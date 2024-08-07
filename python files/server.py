import torch
import logging
import time
import argparse
import os
import psutil
import pandas as pd
import numpy as np
import websockets
import asyncio
from transformers import set_seed, LlamaTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from collections import Counter
import csv
import qlinear
from utils import Utils
from model_utils import (
    warmup,
    decode_prompt,
    get_wikitext2,
    perplexity,
)
from profiler import ProfileAIE
import gc

from modeling_llama_amd import LlamaForCausalLM, LlamaAttention

from pre_quant import run_awq, apply_awq
from quantizer import real_quantize_model_weight
from qmodule import WQLinear

set_seed(123)

class SentenceRetriever:
    def __init__(self, embeddings_file, new_sentences_file, max_tokens=10):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embeddings_file = embeddings_file
        self.new_sentences_file = new_sentences_file
        self.max_tokens = max_tokens
        self.indexes = {}
        self.embeddings_data = self.update_embeddings_with_new_sentences()
        self.sentences = self.extract_sentences()
        self.sentence_counts = Counter(self.sentences)
        self.precompute_indexes()

    def _get_initial_segment(self, sentence, num_tokens):
        tokens = self.tokenizer.tokenize(sentence)
        initial_segment = ' '.join(tokens[:num_tokens])
        return initial_segment

    def update_embeddings_with_new_sentences(self):
        # Load new sentences
        new_sentences = self.load_sentences_from_csv(self.new_sentences_file)

        # Load existing embeddings
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
            print(f"Loaded embeddings from {self.embeddings_file}")
        else:
            embeddings_data = {i: ([], [], []) for i in range(1, self.max_tokens + 1)}
            print(f"Created new embeddings structure")

        # Update embeddings for the new sentences only
        for num_tokens in range(1, self.max_tokens + 1):
            initial_segments_new = [' '.join(self.tokenizer.tokenize(sentence)[:num_tokens]) for sentence in new_sentences]
            embeddings_new = self.model.encode(initial_segments_new)

            existing_embeddings, existing_segments, existing_sentences = embeddings_data[num_tokens]

            updated_embeddings = existing_embeddings + embeddings_new.tolist()
            updated_segments = existing_segments + initial_segments_new
            updated_sentences = existing_sentences + new_sentences

            embeddings_data[num_tokens] = (updated_embeddings, updated_segments, updated_sentences)

            print(f"Updated embeddings for {num_tokens} tokens: {len(updated_embeddings)} entries")

        # Save updated embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
        print(f"Saved updated embeddings to {self.embeddings_file}")

        # Clear new_sentences.csv after updating embeddings
        if os.path.exists(self.new_sentences_file):
            os.remove(self.new_sentences_file)

        return embeddings_data

    def load_sentences_from_csv(self, file_path):
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            sentences = df[0].tolist()
            return sentences
        return []

    def extract_sentences(self):
        all_sentences = []
        for _, _, sentences in self.embeddings_data.values():
            all_sentences.extend(sentences)
        return all_sentences

    def precompute_indexes(self):
        for num_tokens in range(1, self.max_tokens + 1):
            embeddings, initial_segments, _ = self.embeddings_data[num_tokens]
            embeddings = np.array(embeddings)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            self.indexes[num_tokens] = (index, initial_segments)
            print(f"Precomputed index for {num_tokens} tokens.")

    def retrieve(self, query, top_k=5):
        query_tokens = self.tokenizer.tokenize(query)
        num_tokens = len(query_tokens)
        if num_tokens > self.max_tokens:
            num_tokens = self.max_tokens
        query_segment = ' '.join(query_tokens[:num_tokens])
        query_embedding = self.model.encode([query_segment])
        index, initial_segments = self.indexes[num_tokens]
        distances, indices = index.search(query_embedding, top_k * 2)  # Retrieve more to filter by frequency

        candidates = [(self.sentences[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        candidates = sorted(candidates, key=lambda x: (self.sentence_counts[x[0]], -x[1]), reverse=True)
        unique_sentences = []
        seen = set()

        for candidate, _ in candidates:
            if candidate not in seen:
                unique_sentences.append(candidate)
                seen.add(candidate)
            if len(unique_sentences) >= top_k:
                break

        return unique_sentences

async def handle_connection(websocket, path):
    async for message in websocket:
        if "|" in message:
            user_input, prompt_type = message.split("|", 1)
        else:
            user_input = message
            prompt_type = "default"
        print(f"{prompt_type}")
        if prompt_type == "generate_sentence":
            retrieved_sentences = retriever.retrieve(user_input, top_k=10)
            context = "Based on the incomplete sentence and the following similar sentences, complete the sentence.\n\n"
            context += "Similar sentences:\n" + "\n".join(retrieved_sentences) + "\n\n"
            context += "Incomplete sentence: " + user_input + "\n\n"
            context += "Completed sentence:"

            inputs = tokenizer(context, return_tensors="pt")
            outputs = model.generate(inputs.input_ids, max_new_tokens=50)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_lines = response.split('\n')
            final_response = response_lines[-1].strip()
            print(f"Response: {final_response}")
            await websocket.send(final_response)

        elif prompt_type == "generate_from_keywords":
            context = (
                "You are a helpful assistant that helps provide recommendations for completing sentences based on keywords.\n\n"
                "Example 1: Keywords: How day Lisa\n"
                "Response: How is your day, Lisa?\n"
                "Example 2: Keywords: Where dinner tonight\n"
                "Response: Where should we go for dinner tonight?\n\n"
                f"Keywords: {user_input}\n"
                "Response:")
            inputs = tokenizer(context, return_tensors="pt")
            outputs = model.generate(inputs.input_ids, max_new_tokens=50)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_lines = response.split('\n')
            final_response = response_lines[-1].strip()
            print(f"Response: {final_response}")
            await websocket.send(final_response)

        elif prompt_type == "save":

            try:
                with open('new_sentences.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_input])
            except Exception as e:
                print(f"An error occurred: {e}")

        else:
            retrieved_sentences = retriever.retrieve(user_input, top_k=10)
            response = "Retrieved sentences:\n" + "\n".join(retrieved_sentences)
            print(response)
            await websocket.send(response)


def load_model(args):
    tokenizer = LlamaTokenizer.from_pretrained("./llama-2-wts-hf/7B_chat")
    if args.awq == "none":
        model = LlamaForCausalLM.from_pretrained("./llama-2-wts-hf/7B_chat", torch_dtype=torch.bfloat16)

    else:
        ckpt = "pytorch_llama27b_w_bit_{}_awq{}_{}amd.pt".format(args.w_bit, "_fa" if args.flash_attention else "",
                                                                 "lm_" if args.lm_head else "")
        if args.task == "quantize":
            model = LlamaForCausalLM.from_pretrained("./llama-2-wts-hf/7B_chat", torch_dtype=torch.bfloat16)
            print(model)

            Utils.print_model_size(model)

            q_config = {
                "zero_point": True,
                "q_group_size": 128, }  # whether to use group quantization

            if args.awq == 'load':
                print("Loading pre-computed AWQ results from", os.getenv("AWQ_CACHE"))
                awq_results = torch.load(os.getenv("AWQ_CACHE") + "/llama-2-7b-chat-w%d-g128.pt" % args.w_bit,
                                         map_location="cpu")
                apply_awq(model, awq_results)
                print("Quantization config:", q_config)
                real_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )

                Utils.print_model_size(model)

            elif args.awq == 'run':
                awq_results = run_awq(
                    model, tokenizer,
                    w_bit=args.w_bit, q_config=q_config,
                    n_samples=128, seqlen=512,
                )
                torch.save(awq_results, "./llama-2-7b-chat-w%d-g128-generated.pt" % args.w_bit)
                print(model)
                print("Saved AWQ results in ./llama-2-7b-chat-w%d-g128-generated.pt" % args.w_bit)
                raise SystemExit

            if args.flash_attention:
                from llama_flash_attention import LlamaFlashAttention
                node_args = ()
                node_kwargs = {
                    'config': model.config,
                    'llama_name': "llama-2-wts-hf/7B_chat",
                    'flash_config_path': "../../ops/python/llama_flash_attention_config.json",
                    'device': "cpu",  # args.target
                    'max_new_tokens': 11,
                    'quant_mode': "awq"
                }
                Utils.replace_node(model,
                                   LlamaAttention,
                                   LlamaFlashAttention,
                                   node_args, node_kwargs)

            Utils.replace_node(model,
                               WQLinear,
                               qlinear.QLinearPerGrp,
                               (), {'device': 'cpu', 'w_bit': args.w_bit, 'group_size': 128})
            print(model)
            gc.collect()

            Utils.print_model_size(model)
            if args.lm_head:  # Quantize lm_head
                Utils.replace_node(model,
                                   torch.nn.Linear,
                                   qlinear.QLinearPerGrp,
                                   (), {'device': 'cpu', 'w_bit': args.w_bit, 'group_size': 32})
                print(model)
                gc.collect()

            torch.save(model, ckpt)
            print(f"Quantized and saved model: {ckpt}")
            raise SystemExit
        else:
            print(f"Loading from ckpt: {ckpt}")
            if not os.path.exists(ckpt):
                print(
                    f"\n\n ***** Run --task quantize (with/without lm_head) first to save quantized model ...!!! \n\n")
                raise SystemExit
            model = torch.load(ckpt)

    Utils.print_model_size(model)
    _ = gc.collect()
    model.eval()
    model = model.to(torch.bfloat16)
    print(model)
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset - wikitext2-raw-v1, wikitext2-v1", type=str, default="raw",
                        choices=["non-raw", "raw"])
    parser.add_argument('--w_bit', help="weight bit size", type=int, default=3, choices=[3, 4])
    parser.add_argument('--awq', help="load awq scales, clips from pt or run awq", type=str, default="load",
                        choices=["load", "run", "none"])
    parser.add_argument("--target", help="cpu, aie, aie_emu", type=str, default="cpu",
                        choices=["cpu", "aie_emu", "aie"])
    parser.add_argument('--task',
                        help="quantize: Apply AWQ and save ckpt; perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode set of prompts;",
                        type=str, default="decode",
                        choices=["quantize", "decode", "benchmark", "benchmark_long", "perplexity"])
    parser.add_argument('--flash_attention', help="Enable flash attention", action='store_true')
    parser.add_argument('--lm_head', help="Enable PerGrp quantization of lm_head layer", action='store_true')
    parser.add_argument('--num_torch_threads', help="Number of torch threads", type=int, default=8,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    print(f"{args}")
    dev = os.getenv("DEVICE")

    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(args.num_torch_threads)

    log_dir = "./logs_awq_7B_chat"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_awq_7B_chat.log"

    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    model, tokenizer = load_model(args)
    retriever = SentenceRetriever("embeddings.pkl", "new_sentences.csv")

    if args.awq != "none":
        for n, m in model.named_modules():
            if isinstance(m, qlinear.QLinearPerGrp):
                print(f"Preparing weights of layer : {n}")
                m.device = "aie"
                m.quantize_weights()

    print(model)
    Utils.print_model_size(model)

    warmup(model, tokenizer)

    start_server = websockets.serve(handle_connection, "localhost", 8765)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Collections.Generic;

namespace AACwithLLM
{

    [Serializable]
    public class TrieNode
    {
        public Dictionary<char, TrieNode> Children { get; } = new Dictionary<char, TrieNode>();
        public bool IsWord { get; set; }
        public int Count { get; set; } // frequency of each word
    }


    [Serializable]
    public class Trie
    {
        private readonly TrieNode root = new TrieNode();

        public void Insert(string word)
        {
            var node = root;
            foreach (var ch in word)
            {
                if (!node.Children.ContainsKey(ch))
                {
                    node.Children[ch] = new TrieNode();
                }
                node = node.Children[ch];
            }
            node.IsWord = true;
            node.Count += 1; // increment count
        }


        private void DFS(TrieNode node, string prefix, List<string> predictions)
        {


            if (node.IsWord)
            {
                predictions.Add(prefix);
            }

            foreach (var pair in node.Children)
            {
                DFS(pair.Value, prefix + pair.Key, predictions);
            }
        }
        public List<string> Predict(string prefix)
        {
            var predictions = new List<string>();
            var node = root;

            foreach (var ch in prefix)
            {
                if (node.Children.ContainsKey(ch))
                {
                    node = node.Children[ch];
                }
                else
                {
                    return predictions; // Return empty list if prefix is not in trie
                }
            }

            DFS(node, prefix, predictions);
            if (prefix == "") return predictions.OrderByDescending(word => GetNode(word).Count).ToList(); // sort by count
            else return predictions;
        }

        private TrieNode GetNode(string word)
        {
            var node = root;

            foreach (var ch in word)
            {
                if (node.Children.ContainsKey(ch))
                {
                    node = node.Children[ch];
                }
                else
                {
                    return null;
                }
            }

            return node;
        }

    }
    /*
    class Program
    {
        static void Main()
        {
            BinaryFormatter formatter = new BinaryFormatter();
            Trie trie;
            using (FileStream stream = new FileStream("trie.bin", FileMode.Open, FileAccess.Read))
            {
                trie = (Trie)formatter.Deserialize(stream);
            }

            var predictions = trie.Predict("hel");

            foreach (var word in predictions)
            {
                Console.WriteLine(word);
            }
        }
    }
    */

}

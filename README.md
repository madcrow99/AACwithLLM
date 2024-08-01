# AACwithLLM
***Work in progress ***
This project aims to improve the experience of using an AAC program using on-device NLP(Language and speech synthesis models). The interface is a UWP Application with the Windows Eye Control API for optional eye gaze control if appropriate hardware is available.

The following guide describes process of setting up a pc with Windows Eye Control using a compatible eye tracking device. https://support.microsoft.com/en-us/windows/get-started-with-eye-control-in-windows-1a170a20-1083-2452-8f42-17a7d4fe89a9

The following guide describes how to set up visual studio with the required tools for UWPapps. https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/set-up-your-development-environment?tabs=cs-vs-community%2Ccpp-vs-community%2Cvs-2022-17-1-a%2Cvs-2022-17-1-b

The following is a great resource for gaze interaction concepts and how to use Microsoft's Gaze Interaction Library. The requirements near the end of the page need to be included in the project in order to use gaze interaction. https://learn.microsoft.com/en-us/windows/communitytoolkit/gaze/gazeinteractionlibrary

Project Overview:
This project is designed to assist individuals who rely on communication devices to speak, enabling them to communicate more quickly and with less effort. Leveraging the low-power yet high-performance Ryzen AI processor with its built-in Neural Processing Unit (NPU), this application uses a database of the user's input text to predict and display completed sentences as the user types.

Key Features:
1. Real-time Sentence Prediction:The application continually updates a database of the user's input text to predict and display relevant completed sentences as they type.
2. Retrieval-Augmented Generation (RAG): By retrieving relevant sentences from the database, the application reduces the typing effort needed to form desired sentences. Users can select a predicted sentence directly or use it to inform a quantized Llama-2-7B model running on the NPU for a more accurate prediction.
3. Generate Sentence from Keywords: For users who prefer typing less, this function allows them to input a few keywords. The Llama model then returns a well-formed, complete sentence based on those keywords. For example, entering "how day Lisa" might generate responses like "How was your day, Lisa?" or "How are you doing today, Lisa?"

Purpose:
The primary goal of this application is to enhance communication for individuals who have lost their ability to speak. By utilizing on-device computing, the application ensures security and privacy while taking advantage of modern language models to facilitate more natural and effortless communication.

Setup and Initialization:

1. Install Ryzen AI Software and Setup Environment:
   - Follow the instructions provided in the README at [AMD Ryzen AI SW](https://github.com/amd/RyzenAI-SW/tree/1.1/example/transformers/models/llama2).

2. Run the Server Script:
   - Copy `server.py`, `embeddings.pkl`, and `new_sentences.csv` into the folder `example/transformers/models/llama2`.
   - Navigate to the directory containing these files.
   - Run the script using the following command:
     ```bash
     python server.py --task decode --target aie --w_bit 3 --lm_head --flash_attention
     ```

3. Launch the UWP App AACwithLLM:
   - Open Visual Studio and load the solution containing the AACwithLLM project.
   - Build and deploy the AACwithLLM project to your device.
   - Run the AACwithLLM application.

Using the Application:

1. Typing and Prediction:
   - Start typing in the text box. The application will automatically predict and display relevant sentences.
   - You can select any of the predicted sentences to use directly.

2. Generate Sentence from Keywords:
   - Click the "Generate Sentence from Keywords" button.
   - Enter a few keywords related to the sentence you want to form.
   - The application will generate a well-formed sentence based on the provided keywords.

3. Updating the Database:
   - The application automatically updates the database of user input text with each use, improving its prediction accuracy over time.

4. Saving Sentences:
   - Sentences are saved to a file named `new_sentences.csv` each time the "Speak" button is pressed.
   - If the `new_sentences.csv` file already exists, it will be overwritten with the latest sentences.

By integrating the advanced capabilities of the Ryzen AI processor and modern language models, this application represents an advancement in helping individuals communicate more naturally and efficiently.
To do:
-Continue to develop interface
-Fine tune LLM
-Incorporate Text-to-Speech model

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using Windows.Networking.Sockets;
using Windows.Storage.Streams;
using System.Threading.Tasks;
using Windows.System;
using Windows.Media.SpeechSynthesis;
using AACwithLLM;
using Microsoft.Toolkit.Uwp.Input.GazeInteraction;
using System.Text.RegularExpressions;
using System.ServiceModel.Channels;

namespace AACwithLLM
{
    public sealed partial class MainPage : Page
    {
        
        private bool replace_text = true;

        public MainPage()
        {
            this.InitializeComponent();
            WebSocketClient.SetMessageCallback(HandleMessage);
            Connect();
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            Button button = (Button)sender;
            string buttonText = button.Content.ToString() + " ";
            string text = textBox.Text;
            string[] words = text.Split(' ');
            string lastWord = words[words.Length - 1];

            if (button.Name == "MADbutton" | button.Name == "thanksButton") replace_text = false;

            if (replace_text == true) textBox.Text = "";
            else
            {
                // Find the position of the last space character
                int lastSpaceIndex = text.LastIndexOf(' ');

                if (lastSpaceIndex >= 0)
                {
                    // If a space character was found, remove the last word by extracting the text before it
                    string newText = text.Substring(0, lastSpaceIndex);
                    textBox.Text = newText + " ";
                }
                else
                {
                    // If no space character was found, clear the entire TextBox
                    textBox.Text = "";
                }
            }
            textBox.Text += buttonText;
            textBox.Focus(FocusState.Programmatic); // ensure the TextBox has focus
            textBox.Select(textBox.Text.Length, 0); // move the cursor to the end of the TextBox

        }
        private async void ButtonSpeak_Click(object sender, RoutedEventArgs e)
        {
            string text = textBox.Text;
            text = text.Trim();
            string promptType = "";
            if (text.Length > 0)
            {
                SpeechSynthesizer synthesizer = new SpeechSynthesizer();
                SpeechSynthesisStream synthesisStream = await synthesizer.SynthesizeTextToStreamAsync(text);
                mediaElement.SetSource(synthesisStream, synthesisStream.ContentType);
                mediaElement.Play();
                textBox.Text = "";
                Button button = (Button)FindName("button2");
                button.Content = text;

                // Send save command to Python server to update csv
                promptType = "save";
                SendMessage($"{text}|{promptType}");
                
            }
        }

        private async void Generate_Click(object sender, RoutedEventArgs e)
        {
            Button button = (Button)sender;
            string buttonText = button.Content.ToString();
            string text = textBox.Text;
            string message = text.Trim();
            string promptType = "";

            if (button.Name == "GenerateSentenceButton")
            {
                promptType = "generate_sentence";
            }
            else if (button.Name == "GenerateFromKeywordsButton")
            {
                promptType = "generate_from_keywords";
            }

            if (!string.IsNullOrWhiteSpace(promptType))
            {
                ClearButtons(); // Clear button contents
                SendMessage($"{message}|{promptType}");
            }

            textBox.Focus(FocusState.Programmatic);
            textBox.Select(textBox.Text.Length, 0);
        }

        private void ClearButtons()
        {
            Button[] buttons = { button2, button3, button4, button5, button6, button7, button8, button9, button10, button11, button12, button13, button14, button15, button16 };
            foreach (var button in buttons)
            {
                button.Content = "";
            }
        }

        private async void HandleMessage(string message)
        {
            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, () =>
            {
                UpdateButtons(message);
            });
        }

        private void UpdateButtons(string responseData)
        {
            Button[] buttons = { button2, button3, button4, button5, button6, button7, button8, button9, button10, button11, button12, button13, button14, button15, button16 };

            // Clear all button contents first
            foreach (var button in buttons)
            {
                button.Content = "";
            }

            // Determine the type of response based on the presence of certain keywords
            if (responseData.StartsWith("Retrieved sentences:"))
            {
                var retrievedSentences = responseData.Split(new[] { "\n" }, StringSplitOptions.RemoveEmptyEntries).Skip(1);
                int i = 0;
                foreach (var sentence in retrievedSentences)
                {
                    if (i < buttons.Length)
                    {
                        buttons[i].Content = sentence;
                        i++;
                    }
                }
            }
            else
            {
                // For generate sentence from keywords
                // Split responseData by ": " and take the last part, assuming it is the final generated sentence.
                var responseLines = responseData.Split(new[] { ": " }, StringSplitOptions.RemoveEmptyEntries);
                if (responseLines.Length > 1)
                {
                    var finalSentence = responseLines[responseLines.Length - 1].Trim();
                    if (!string.IsNullOrWhiteSpace(finalSentence))
                    {
                        button2.Content = finalSentence; // Assuming you display the final sentence in the first button.
                    }
                }
            }
        }


        private string previousText = "   ";

        private void TextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            TextBox textBox = sender as TextBox;
            if (textBox != null)
            {
                string currentText = textBox.Text;

                if (currentText.Length >= 3 && (
                        currentText.Substring(0, 3) != previousText.Substring(0, 3) ||
                        Math.Abs(currentText.Length - previousText.Length) > 3)
                   )
                {
                    SendMessage($"{currentText}|default");
                    previousText = currentText;
                }
            }
        }

        private void TextBox_KeyDown(object sender, KeyRoutedEventArgs e)
        {
            if (e.Key == VirtualKey.Enter)
            {
                e.Handled = true;
                ButtonSpeak_Click(sender, e);
            }
        }

        private async void Connect()
        {
            string serverUrl = "ws://localhost:8765";
            try
            {
                await WebSocketClient.ConnectAsync(serverUrl);
            }
            catch (Exception ex)
            {
                // Handle the exception
                Console.WriteLine($"Error connecting to server: {ex.Message}");
            }
        }

        private async void SendMessage(string message)
        {
            try
            {
                await WebSocketClient.SendMessageAsync(message);
            }
            catch (Exception ex)
            {
                // Handle the exception
                Console.WriteLine($"Error sending message: {ex.Message}");
            }
        }




        protected override async void OnNavigatedFrom(NavigationEventArgs e)
        {
            // Disconnect the WebSocket client when navigating away from the page
            await WebSocketClient.CloseAsync();
            base.OnNavigatedFrom(e);
        }

    }
}


using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace AACwithLLM
{

    public static class WebSocketClient
    {
        private static ClientWebSocket _webSocket;
        private static Action<string> _messageCallback;

        public static async Task ConnectAsync(string serverUrl)
        {
            Uri serverUri = new Uri(serverUrl);
            _webSocket = new ClientWebSocket();
            await _webSocket.ConnectAsync(serverUri, CancellationToken.None);

            // Start a background task to receive messages from the WebSocket server
            _ = ReceiveMessagesAsync();
        }

        public static async Task SendMessageAsync(string message)
        {
            byte[] messageBytes = Encoding.UTF8.GetBytes(message);
            await _webSocket.SendAsync(new ArraySegment<byte>(messageBytes), WebSocketMessageType.Text, true, CancellationToken.None);
        }

        public static void SetMessageCallback(Action<string> messageCallback)
        {
            _messageCallback = messageCallback;
        }

        public static async Task CloseAsync()
        {
            await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", CancellationToken.None);
        }

        private static async Task ReceiveMessagesAsync()
        {
            byte[] receiveBuffer = new byte[1024];

            try
            {
                while (_webSocket.State == WebSocketState.Open)
                {
                    WebSocketReceiveResult receiveResult = await _webSocket.ReceiveAsync(new ArraySegment<byte>(receiveBuffer), CancellationToken.None);
                    string receivedMessage = Encoding.UTF8.GetString(receiveBuffer, 0, receiveResult.Count);

                    // Call the message callback function with the received message
                    _messageCallback?.Invoke(receivedMessage);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error receiving message: {ex.Message}");
            }
        }
    }
}

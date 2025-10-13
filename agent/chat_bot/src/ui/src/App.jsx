
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    {
      text: 'Hello! I\' am a chat bot that know a lot about top scientists in history. Feel free to ask me anything about them.',
      sender: 'bot'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [tickets, setTickets] = useState([]);

  const fetchTickets = async () => {
    try {
      const response = await fetch('http://localhost:8000/tickets');
      const data = await response.json();
      setTickets(data);
    } catch (error) {
      console.error('Error fetching tickets:', error);
    }
  };

  useEffect(() => {
    fetchTickets();
  }, []);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      const newMessages = [...messages, { text: inputValue, sender: 'user' }];
      setMessages(newMessages);
      setInputValue('');

      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(newMessages),
        });
        const data = await response.json();
        setMessages(prevMessages => [...prevMessages, { text: data.response, sender: 'bot' }]);
        fetchTickets(); // Refetch tickets after bot response
      } catch (error) {
        console.error('Error fetching data:', error);
        setMessages(prevMessages => [...prevMessages, { text: 'Error connecting to the server.', sender: 'bot' }]);
      }
    }
  };

  return (
    <div className="app-container">
      <div className="ticket-list">
        <h2>Tickets</h2>
        {tickets.map(ticket => (
          <div key={ticket.id} className="ticket">
            <p><b>ID:</b> {ticket.id}</p>
            <p><b>Timestamp:</b> {new Date(ticket.timestamp).toLocaleString()}</p>
            <p><b>Question:</b> {ticket.user_question}</p>
            <p><b>Error:</b> {ticket.error_description || 'N/A'}</p>
            <p><b>Status:</b> {ticket.status}</p>
            <div className="ticket-conversation">
              <b>Conversation:</b>
              {ticket.conversation_history ? (
                JSON.parse(ticket.conversation_history).map((msg, i) => (
                  <div key={i} className={`message-sm ${msg.sender}`}>
                    <span><b>{msg.sender}:</b> {msg.text}</span>
                  </div>
                ))
              ) : (
                <p>No history available.</p>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="chat-container">
        <div className="message-list">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              {message.text}
            </div>
          ))}
        </div>
        <form className="message-form" onSubmit={handleSendMessage}>
          <input
            type="text"
            className="message-input"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your message..."
          />
          <button type="submit" className="send-button">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;

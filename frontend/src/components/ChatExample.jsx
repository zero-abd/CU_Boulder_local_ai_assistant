import { useEffect, useState } from 'react';
import buffAdvisorApi from '../api/buffAdvisor';
import './ChatExample.css';

// Example component showing how to use the BuffAdvisor API with voice capabilities
export default function ChatExample() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [style, setStyle] = useState('balanced');
  
  // Speech recognition setup
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = SpeechRecognition ? new SpeechRecognition() : null;
  
  useEffect(() => {
    if (recognition) {
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      
      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');
        
        setMessage(transcript);
      };
      
      recognition.onend = () => {
        setIsListening(false);
      };
    }
  }, [recognition]);
  
  // Text-to-speech setup
  const speak = (text) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };
  
  // Start or stop listening
  const toggleListening = () => {
    if (isListening) {
      recognition?.stop();
      setIsListening(false);
    } else {
      recognition?.start();
      setIsListening(true);
    }
  };
  
  // Send message to BuffAdvisor
  const sendMessage = async () => {
    if (!message.trim()) return;
    
    setIsLoading(true);
    try {
      const result = await buffAdvisorApi.sendMessage(message, style);
      setResponse(result.response);
      speak(result.response);
    } catch (error) {
      console.error('Error sending message:', error);
      setResponse('Sorry, there was an error processing your request.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Check if backend is ready
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const isReady = await buffAdvisorApi.checkStatus();
        setBackendReady(isReady);
        
        // If not ready, keep checking every 5 seconds
        if (!isReady) {
          setTimeout(checkBackendStatus, 5000);
        }
      } catch (error) {
        console.error('Error checking backend status:', error);
        setTimeout(checkBackendStatus, 5000);
      }
    };
    
    checkBackendStatus();
    
    // Cleanup function
    return () => {
      if (isListening && recognition) {
        recognition.stop();
      }
    };
  }, []);
  
  return (
    <div className="chat-container">
      <div className="status-indicator">
        Backend Status: {backendReady ? 'Ready' : 'Initializing...'}
      </div>
      
      <div className="response-area">
        {response ? response : 'Ask BuffAdvisor a question about CU Boulder!'}
      </div>
      
      <div className="input-area">
        <select 
          value={style} 
          onChange={(e) => setStyle(e.target.value)}
          className="style-selector"
        >
          <option value="balanced">Balanced</option>
          <option value="brief">Brief</option>
          <option value="detailed">Detailed</option>
          <option value="supportive">Supportive</option>
        </select>
        
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your question here..."
          disabled={isListening}
          className="message-input"
        />
        
        <div className="button-group">
          <button 
            onClick={toggleListening} 
            className={`voice-button ${isListening ? 'active' : ''}`}
            disabled={!recognition}
          >
            {isListening ? 'Stop Listening' : 'Start Voice Input'}
          </button>
          
          <button 
            onClick={sendMessage} 
            disabled={!message.trim() || isLoading || !backendReady}
            className="send-button"
          >
            {isLoading ? 'Loading...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
} 
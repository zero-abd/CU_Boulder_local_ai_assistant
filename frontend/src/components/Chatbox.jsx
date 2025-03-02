import React, { useEffect, useRef, useState } from 'react';
import buffAdvisorApi from '../api/buffAdvisor';
// Import CU Boulder logo
import buffLogo from '../assets/cu-boulder-logo.svg';

const Chatbox = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [timeoutId, setTimeoutId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  const [solutionStyle, setSolutionStyle] = useState('brief');
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceDetectionActive, setFaceDetectionActive] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Declare SpeechRecognition instance
  const [recognition, setRecognition] = useState(null);

  // Variables to track active requests for cancellation
  const [activeRequest, setActiveRequest] = useState(null);

  // Check API status on component mount
  useEffect(() => {
    let isMounted = true;
    const retryDelay = 2000; // 2 seconds between retries
    let retryCount = 0;
    const maxRetries = 5;

    const checkApiStatus = async () => {
      if (!isMounted) return;
      
      try {
        console.log(`API check attempt ${retryCount + 1}/${maxRetries || 'unlimited'}`);
        const isHealthy = await buffAdvisorApi.healthCheck();
        
        if (isHealthy) {
          const isReady = await buffAdvisorApi.checkStatus();
          if (isMounted) {
            setApiReady(isReady);
            console.log('API status:', isReady ? 'ready' : 'initializing');
            
            // Reset retry count if we got a successful response
            retryCount = 0;
          }
        } else {
          console.error('API health check failed');
          if (isMounted && retryCount < maxRetries) {
            retryCount++;
            setTimeout(checkApiStatus, retryDelay);
          }
        }
      } catch (error) {
        console.error('Error checking API status:', error);
        if (isMounted && retryCount < maxRetries) {
          retryCount++;
          setTimeout(checkApiStatus, retryDelay);
        }
      }
    };
    
    // Initial check
    checkApiStatus();
    
    // Poll API status every 5 seconds if not ready
    const interval = setInterval(() => {
      if (isMounted && !apiReady) {
        checkApiStatus();
      }
    }, 5000);
    
    // Cleanup
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [apiReady]);

  useEffect(() => {
    // Initialize SpeechRecognition only once
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      setRecognition(recognitionInstance);
    }

    loadFaceDetectionModels();
  }, []);

  const loadFaceDetectionModels = async () => {
    try {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js';
      script.async = true;
      script.onload = () => {
        initFaceDetection();
      };
      document.head.appendChild(script);
    } catch (error) {
      console.error('Error loading face detection models:', error);
    }
  };

  const initFaceDetection = async () => {
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models')
      ]);
      
      console.log('Face detection models loaded');
      setFaceDetectionActive(true);
      startVideo();
    } catch (error) {
      console.error('Error initializing face detection:', error);
    }
  };

  const startVideo = async () => {
    try {
      if (streamRef.current) return;
      
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      detectFace();
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const detectFace = () => {
    if (!videoRef.current || !faceapi || !faceDetectionActive) return;

    const timer = setInterval(async () => {
      if (!videoRef.current) {
        clearInterval(timer);
        return;
      }

      try {
        const detections = await faceapi.detectAllFaces(
          videoRef.current, 
          new faceapi.TinyFaceDetectorOptions()
        );

        if (detections.length > 0 && !faceDetected) {
          console.log('Face detected!');
          setFaceDetected(true);
          speakWelcomeMessage();
          clearInterval(timer);
        }
      } catch (error) {
        console.error('Face detection error:', error);
      }
    }, 1000);
  };

  const speakWelcomeMessage = () => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance("Hello, how can I assist you?");
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };

  const stopVideo = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  // Scroll to bottom when messages update
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (recognition) {
      recognition.continuous = true; 
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        console.log('Speech recognition started');
        setIsListening(true);
        // Clear any previous timeout when restarting speech recognition
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
      };

      recognition.onend = () => {
        console.log('Speech recognition ended');
        setIsListening(false);
      };

      recognition.onresult = async (event) => {
        // Only process speech results if we're not already loading a response
        if (!isLoading) {
          const transcript = event.results[event.results.length - 1][0].transcript;
          console.log('Speech recognized:', transcript);
          
          // Add user message
          addMessage(transcript, 'user');

          // Get response from API
          await getAIResponse(transcript);
        } else {
          console.log('Speech result ignored - system is busy');
          // Force stop listening since system is busy
          stopListening();
        }

        // Auto-stop after a few seconds of silence
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        const newTimeoutId = setTimeout(() => {
          recognition.stop();
        }, 3000);
        setTimeoutId(newTimeoutId);
      };

      recognition.onerror = (event) => {
        console.error('Speech Recognition Error:', event);
        setIsListening(false);
      };
    }
  }, [recognition, timeoutId, isLoading]);

  useEffect(() => {
    // Automatically stop listening whenever isLoading changes to true
    if (isLoading && isListening) {
      console.log('Auto-stopping microphone because system is now busy');
      stopListening();
    }
  }, [isLoading]);

  useEffect(() => {
    if (faceDetected && !isListening && !isLoading && recognition) {
      setTimeout(() => {
        startListening();
      }, 1000);
    }
  }, [faceDetected, isListening, isLoading, recognition]);

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  const startListening = () => {
    if (recognition && !isLoading) {
      try {
        recognition.start();
        console.log('Starting voice recognition');
      } catch (error) {
        console.error('Failed to start recognition:', error);
        setIsListening(false);
      }
    } else if (isLoading) {
      console.log('Cannot start listening while processing a response');
    } else {
      console.error('Speech recognition not available');
    }
  };

  const stopListening = () => {
    console.log('Stopping speech recognition');
    if (recognition) {
      try {
        recognition.stop();
      } catch (error) {
        console.error('Error stopping recognition:', error);
      }
    }
    if (timeoutId) {
      clearTimeout(timeoutId);
      setTimeoutId(null);
    }
    setIsListening(false);
  };

  const addMessage = (text, sender, id = Date.now() + '-' + Math.random().toString(36).substring(2, 9), isThinking = false) => {
    const message = {
      id,
      text,
      sender,
      isThinking,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(messages => [...messages, message]);
  };

  // Get AI response to user message
  const getAIResponse = async (userMessage) => {
    // Add a "thinking" message first (will be replaced with actual response)
    const thinkingId = Date.now() + '-' + Math.random().toString(36).substring(2, 9);
    addMessage("thinking...", "assistant", thinkingId, true);
    setIsLoading(true);
    
    let streamingConnection = null;
    let streamingResponse = "";
    let streamingMessageId = null;
    
    try {
      // Use streaming API for faster response display
      streamingMessageId = Date.now() + '-' + Math.random().toString(36).substring(2, 9);
      
      // Start with empty message that will be updated as chunks arrive
      addMessage("", "assistant", streamingMessageId);
      
      streamingConnection = buffAdvisorApi.sendMessageStream(
        userMessage, 
        solutionStyle,
        false, // new session
        // Callback for each chunk
        (chunk) => {
          streamingResponse += chunk;
          updateMessage(streamingMessageId, streamingResponse);
        },
        // Callback for completion
        (data) => {
          console.log('Streaming complete:', data);
          setIsLoading(false);
          // Remove the thinking message now that we have a real response
          removeMessage(thinkingId);
        },
        // Callback for errors
        (error) => {
          console.error('Streaming error:', error);
          setIsLoading(false);
          
          // Remove the thinking message
          removeMessage(thinkingId);
          
          // Add the error message
          if (streamingResponse) {
            // If we got some response before the error, keep it
            updateMessage(streamingMessageId, streamingResponse + "\n\n[Connection lost. Response may be incomplete.]");
          } else {
            // If no response was received, show a full error message
            removeMessage(streamingMessageId);
            const errorMessage = error.friendlyMessage || "Failed to get response from the assistant. Please try again.";
            addMessage(errorMessage, "error");
          }
        }
      );
      
      // Store the connection for potential cancellation
      setActiveRequest(streamingConnection);
      
    } catch (error) {
      // This shouldn't normally happen since errors should be caught in the error callback
      console.error('Unexpected error during streaming setup:', error);
      removeMessage(thinkingId);
      if (streamingMessageId) removeMessage(streamingMessageId);
      
      let errorMessage = "An error occurred while getting the response.";
      
      if (error.isTimeout) {
        errorMessage = "The request timed out. Please try a simpler question or try again later.";
      } else if (error.message && error.message.includes("abort")) {
        errorMessage = "Request was cancelled.";
      } else if (error.friendlyMessage) {
        errorMessage = error.friendlyMessage;
      }
      
      addMessage(errorMessage, "error");
      setIsLoading(false);
    }
    
    // Return the connection object with abort method
    return {
      abort: () => {
        if (streamingConnection) {
          streamingConnection.abort();
          setIsLoading(false);
          removeMessage(thinkingId);
          if (streamingResponse) {
            updateMessage(streamingMessageId, streamingResponse + "\n\n[Response cancelled by user]");
          } else if (streamingMessageId) {
            removeMessage(streamingMessageId);
          }
        }
      }
    };
  };

  // Function to update an existing message by id
  const updateMessage = (id, content) => {
    setMessages(prevMessages => 
      prevMessages.map(msg => 
        msg.id === id ? { ...msg, text: content } : msg
      )
    );
  };
  
  // Function to remove a message by id
  const removeMessage = (id) => {
    setMessages(prevMessages => 
      prevMessages.filter(msg => msg.id !== id)
    );
  };

  // Function to cancel active request
  const cancelActiveRequest = () => {
    if (activeRequest) {
      console.log('Cancelling active request');
      activeRequest.abort();
      setActiveRequest(null);
    }
  };

  // Function to handle message submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (isLoading) {
      // If we're already loading, the button acts as a cancel button
      cancelActiveRequest();
      return;
    }
    
    if (!inputText.trim()) return;
    
    // Stop listening if active
    if (isListening) {
      try {
        recognition.stop();
        setIsListening(false);
      } catch (error) {
        console.error('Error stopping recognition:', error);
      }
    }
    
    // Store the message text before clearing it
    const userMessage = inputText;
    
    // Add user message to chat and clear input
    addMessage(userMessage, 'user');
    setInputText('');
    
    // Get AI response and store the request for potential cancellation
    const request = await getAIResponse(userMessage);
    setActiveRequest(request);
  };

  // Add the Material Icons stylesheet in the component
  useEffect(() => {
    // Add Material Icons stylesheet
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/icon?family=Material+Icons';
    document.head.appendChild(link);
    
    return () => {
      // Clean up on component unmount
      document.head.removeChild(link);
    };
  }, []);

  useEffect(() => {
    return () => {
      stopVideo();
      if (recognition) {
        try {
          recognition.stop();
        } catch (error) {
          console.error('Error stopping recognition on unmount:', error);
        }
      }
    };
  }, []);

  return (
    <div className="chatbox">
      <div className="chat-header">
        <div className="chat-title">
          <img src={buffLogo} alt="BuffAdvisor Logo" className="app-logo" />
          <h2>BuffAdvisor</h2>
        </div>
        <div className="chat-controls">
          <select 
            value={solutionStyle} 
            onChange={(e) => setSolutionStyle(e.target.value)}
            className="solution-style-selector"
          >
            <option value="brief">Brief</option>
            <option value="detailed">Detailed</option>
            <option value="supportive">Supportive</option>
            <option value="balanced">Balanced</option>
          </select>
        </div>
      </div>
      
      <div className={`video-container ${faceDetected ? 'minimized' : ''}`}>
        <video 
          ref={videoRef}
          autoPlay
          muted
          playsInline
          width="320"
          height="240"
          className="face-detection-video"
        />
        {faceDetectionActive && !faceDetected && <div className="scanning-indicator">Scanning for face...</div>}
        {faceDetected && <div className="face-detected-indicator">Face detected</div>}
      </div>
      
      <div className="chat-messages" ref={chatContainerRef}>
        {messages.map((message) => (
          <div 
            key={message.id} 
            className={`message ${message.sender === 'user' ? 'user-message' : 'ai-message'} ${message.isThinking ? 'thinking' : ''}`}
          >
            <div className="message-content">
              {message.isThinking ? (
                <div className="thinking-animation">
                  <span>●</span><span>●</span><span>●</span>
                </div>
              ) : (
                <div 
                  className={message.sender === 'error' ? 'error-message' : ''}
                >
                  {message.text}
                </div>
              )}
            </div>
            <div className="message-time">{message.timestamp}</div>
          </div>
        ))}
        {messages.length === 0 && (
          <div className="empty-chat">
            <div className="welcome-message">
              <h3>Welcome to BuffAdvisor!</h3>
              <p>Your AI assistant for University of Colorado Boulder information.</p>
              <p>How can I help you today?</p>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask a question about CU Boulder..."
          disabled={isLoading && !activeRequest}
        />
        <div className="input-buttons">
          <button 
            type="button" 
            onClick={toggleListening}
            className={`mic-button ${isListening ? 'active' : ''}`}
            disabled={isLoading && !activeRequest}
          >
            <span className="material-icons">
              {isListening ? 'mic' : 'mic_none'}
            </span>
          </button>
          <button 
            type="submit" 
            className={`submit-button ${isLoading ? 'cancel-button' : ''}`}
            disabled={isLoading && !activeRequest}
          >
            <span className="material-icons">
              {isLoading ? 'cancel' : 'send'}
            </span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default Chatbox;

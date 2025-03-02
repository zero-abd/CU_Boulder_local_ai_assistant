import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  }
});

apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      console.error('API Error Response:', {
        status: error.response.status,
        data: error.response.data,
        headers: error.response.headers
      });
    } else if (error.request) {
      console.error('API Request Error (No Response):', error.request);
    } else {
      console.error('API Error:', error.message);
    }
    return Promise.reject(error);
  }
);

const buffAdvisorApi = {
  async checkStatus() {
    try {
      console.log('Checking API status at:', `${API_URL}/status`);
      const response = await apiClient.get('/status');
      console.log('Status response:', response.data);
      return response.data.ready;
    } catch (error) {
      console.error('Error checking status:', error);
      return false;
    }
  },

  sendMessageStream(message, style = 'brief', newSession = false, onChunk, onComplete, onError) {
    console.log('Setting up streaming message to backend API:', { message, style, new_session: newSession });
    
    const controller = new AbortController();
    
    apiClient.post('/chat', {
      message,
      style,
      new_session: newSession,
      streaming: true
    }, { signal: controller.signal })
      .then(response => {
        console.warn('Received non-streaming response:', response.data);
        if (onComplete) onComplete(response.data);
      })
      .catch(error => {
        console.error('Error setting up streaming response:', error);
        if (onError) onError(error);
      });
    
    const eventSource = new EventSource(`${API_URL}/chat?message=${encodeURIComponent(message)}&style=${style}&new_session=${newSession}&streaming=true`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.status === 'generating' && data.chunk) {
          if (onChunk) onChunk(data.chunk);
        } 
        else if (data.status === 'complete') {
          console.log('Streaming completed:', data);
          eventSource.close();
          if (onComplete) onComplete(data);
        }
        else if (data.status === 'error') {
          console.error('Streaming error:', data.message);
          eventSource.close();
          if (onError) onError(new Error(data.message));
        }
      } catch (error) {
        console.error('Error parsing SSE data:', error, event.data);
        if (onError) onError(error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
      if (onError) onError(new Error('Connection to server lost'));
    };
    
    return {
      abort: () => {
        console.log('Aborting streaming connection');
        controller.abort();
        eventSource.close();
      }
    };
  },

  async sendMessage(message, style = 'balanced', newSession = false, options = {}) {
    try {
      console.log('Sending message to backend API:', { message, style, new_session: newSession });
      console.log('API URL:', API_URL);
      
      const requestConfig = {
        ...options
      };
      
      const response = await apiClient.post('/chat', {
        message,
        style,
        new_session: newSession,
        streaming: false
      }, requestConfig);
      
      console.log('API response received:', response.data);
      
      if (!response.data.response) {
        console.warn('Response does not contain expected "response" field:', response.data);
      }
      
      return response.data;
    } catch (error) {
      console.error('Error sending message to backend:', error);
      
      if (error.code === 'ECONNABORTED') {
        error.isTimeout = true;
        error.friendlyMessage = 'Request timed out. The server might be busy or experiencing issues.';
      } else if (error.response) {
        error.friendlyMessage = error.response.data?.message || 
                               error.response.data?.error || 
                               `Server error (${error.response.status})`;
      } else if (error.request) {
        error.friendlyMessage = 'No response received from server. Please check your connection.';
      } else {
        error.friendlyMessage = 'Error setting up request. Please try again.';
      }
      
      throw error;
    }
  },

  async healthCheck() {
    try {
      console.log('Performing health check at:', `${API_URL}/health`);
      const response = await apiClient.get('/health', { timeout: 5000 });
      console.log('Health check response:', response.data);
      return true;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
};

export default buffAdvisorApi;
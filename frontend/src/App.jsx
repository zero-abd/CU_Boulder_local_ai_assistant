import React from 'react';
import './App.css';
import Chatbox from './components/Chatbox';
import Navbar from './components/Navbar';
import WebcamComponent from './components/WebcamComponent';

const App = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 relative overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 pattern-grid opacity-5 pointer-events-none"></div>
      
      <Navbar />

      {/* Main content container with improved spacing */}
      <div className="container mx-auto px-4 pt-24 pb-6 h-[calc(100vh-2rem)]">
        <div className="flex flex-col md:flex-row gap-6 h-full">
          
          {/* Chatbox Section - Enhanced with better styling */}
          <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-2xl w-full md:w-1/2 overflow-hidden border border-gray-700 transition-all duration-300 flex flex-col">
            <Chatbox />
          </div>

          {/* Webcam Section - Enhanced with better styling */}
          <div className="w-full md:w-1/2 bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-2xl border border-gray-700 p-4 flex items-center justify-center transition-all duration-300">
            <WebcamComponent />
          </div>
        </div>
      </div>

      <footer>
        <p>Powered by local LLMs and the Web Speech API</p>
      </footer>
    </div>
  );
};

export default App;

import React, { useState } from 'react';
import Webcam from "react-webcam";

const WebcamComponent = () => {
  const [isFullscreen, setIsFullscreen] = useState(false);

  return (
    <div className="w-full h-full flex flex-col items-center justify-center">
      <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-gray-900/90 flex items-center justify-center' : ''}`}>
        <Webcam 
          width={isFullscreen ? 'auto' : null}
          height={isFullscreen ? 'auto' : null}
          className={`
            border-2 border-indigo-500/50 rounded-xl shadow-lg
            ${isFullscreen ? 'max-h-[90vh] max-w-[90vw]' : 'w-full max-w-md'}
            transition-all duration-300 object-cover
          `}
        />
        
        {/* Controls overlay */}
        <div className="absolute bottom-3 right-3 flex space-x-2">
          <button 
            onClick={() => setIsFullscreen(!isFullscreen)} 
            className="bg-gray-800/80 hover:bg-gray-700/80 text-white p-2 rounded-full backdrop-blur-sm transition-colors"
          >
            {isFullscreen ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M5 9V7a2 2 0 012-2h2V3H7a4 4 0 00-4 4v2h2zm10-2V5a2 2 0 00-2-2h-2V1h2a4 4 0 014 4v2h-2zM5 11v2a2 2 0 002 2h2v2H7a4 4 0 01-4-4v-2h2zm10 4v-2a2 2 0 00-2-2h-2v2h2a2 2 0 002 2v2a4 4 0 01-4-4v-2h4v2z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3 4a1 1 0 011-1h4a1 1 0 010 2H6.414l2.293 2.293a1 1 0 01-1.414 1.414L5 6.414V8a1 1 0 01-2 0V4zm9 1a1 1 0 110-2h4a1 1 0 011 1v4a1 1 0 11-2 0V6.414l-2.293 2.293a1 1 0 11-1.414-1.414L13.586 5H12zm-9 7a1 1 0 112 0v1.586l2.293-2.293a1 1 0 011.414 1.414L6.414 15H8a1 1 0 110 2H4a1 1 0 01-1-1v-4zm13-1a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 110-2h1.586l-2.293-2.293a1 1 0 011.414-1.414L15 13.586V12a1 1 0 011-1z" clipRule="evenodd" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default WebcamComponent;

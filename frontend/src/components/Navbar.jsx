import React, { useState } from 'react';

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="text-white fixed top-0 left-0 right-0 z-50 bg-gray-900/90 backdrop-blur-md border-b border-gray-800 shadow-lg transition-all duration-300">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        {/* Logo area */}
        <div className="flex items-center space-x-2">
          <div className="h-8 w-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg">
            <span className="text-xl font-bold">H</span>
          </div>
          <span className="text-xl md:text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
            HackCU 2025
          </span>
        </div>

        {/* Desktop Menu - hidden on mobile */}
        <div className="hidden md:flex items-center space-x-6">
          <a href="#" className="text-gray-300 hover:text-white transition-colors">Home</a>
          <a href="#" className="text-gray-300 hover:text-white transition-colors">About</a>
          <a href="#" className="text-gray-300 hover:text-white transition-colors">Contact</a>
        </div>

        {/* Mobile menu button */}
        <button 
          onClick={() => setMenuOpen(!menuOpen)}
          className="md:hidden text-gray-300 hover:text-white focus:outline-none"
        >
          <svg className="w-6 h-6" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
            {menuOpen ? (
              <path d="M6 18L18 6M6 6l12 12"></path>
            ) : (
              <path d="M4 6h16M4 12h16M4 18h16"></path>
            )}
          </svg>
        </button>
      </div>

      {/* Mobile Menu - Slides down when open */}
      <div className={`md:hidden bg-gray-800 transition-all duration-300 overflow-hidden ${menuOpen ? 'max-h-56 border-t border-gray-700' : 'max-h-0'}`}>
        <div className="container mx-auto px-4 py-2">
          <a href="#" className="block py-2 text-gray-300 hover:text-white transition-colors">Home</a>
          <a href="#" className="block py-2 text-gray-300 hover:text-white transition-colors">About</a>
          <a href="#" className="block py-2 text-gray-300 hover:text-white transition-colors">Contact</a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

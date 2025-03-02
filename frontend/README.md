# BuffAdvisor Frontend

This is the frontend for the BuffAdvisor application, powered by React and Vite.

## Setup and Running

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file in the frontend directory with:
   ```
   VITE_API_URL=http://localhost:5000/api
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will be available at http://localhost:5173.

## Project Structure

- `src/components/`: UI components including ChatExample with voice capabilities
- `src/api/`: API client for communicating with the Flask backend
- `src/App.jsx`: Main application component

## Features

- Voice input via Web Speech API
- Text-to-speech for responses
- Different response styles (balanced, brief, detailed, supportive)
- Status indicators for backend connection

## Troubleshooting

- If you encounter errors related to missing dependencies, run `npm install` again
- Make sure the backend is running and available at http://localhost:5000
- Check the console for any API connection errors

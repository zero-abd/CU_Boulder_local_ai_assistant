from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import time
import traceback
import threading
import bot

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"], "methods": ["GET", "POST", "OPTIONS"]}})

initialized = False
initialization_lock = threading.Lock()
request_threads = {}
server_metrics = {
    "start_time": time.time(),
    "requests_processed": 0,
    "timeouts": 0,
    "errors": 0,
    "avg_response_time": 0,
    "total_response_time": 0
}

def initialize_buffadvisor():
    global initialized
    
    with initialization_lock:
        if initialized:
            print("[DEBUG] BuffAdvisor already initialized, skipping")
            return True
            
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
        if pdf_files:
            pdf_path = pdf_files[0]
            print(f"[DEBUG] Initializing with PDF: {pdf_path}")
            try:
                bot.initialize_advisor(pdf_path)
                if bot.advisor_instance is not None:
                    initialized = True
                    return True
                else:
                    print("[ERROR] Initialization failed - advisor_instance is None")
                    return False
            except Exception as e:
                print(f"[ERROR] Exception during initialization: {str(e)}")
                traceback.print_exc()
                return False
        else:
            print("[ERROR] No PDF files found in the current directory.")
            print("Please place a CU Boulder information PDF file in the current directory.")
            return False

def cleanup_request_threads():
    global request_threads
    current_time = time.time()
    
    stale_thread_ids = []
    for thread_id, thread_info in request_threads.items():
        if current_time - thread_info["start_time"] > 300:
            stale_thread_ids.append(thread_id)
    
    for thread_id in stale_thread_ids:
        del request_threads[thread_id]
    
    print(f"[DEBUG] Cleaned up {len(stale_thread_ids)} stale request threads")


@app.errorhandler(500)
def handle_500_error(e):
    global server_metrics
    server_metrics["errors"] += 1
    print(f"[ERROR] 500 error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(e),
        'status': 'error'
    }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    global server_metrics
    print("[DEBUG] Health check requested")
    
    cleanup_request_threads()
    
    uptime = time.time() - server_metrics["start_time"]
    
    response = {
        'status': 'ok',
        'uptime': f"{uptime:.2f} seconds",
        'uptime_hours': f"{uptime/3600:.2f} hours",
        'requests_processed': server_metrics["requests_processed"],
        'timeouts': server_metrics["timeouts"],
        'errors': server_metrics["errors"],
        'active_threads': len(request_threads),
        'avg_response_time': f"{server_metrics['avg_response_time']:.2f} seconds" if server_metrics["requests_processed"] > 0 else "N/A"
    }
    
    return jsonify(response), 200


@app.route('/api/status', methods=['GET'])
def check_status():
    global initialized
    print("[DEBUG] Status check requested")
    if not initialized and bot.advisor_instance is None:
        print("[DEBUG] Initializing advisor in status check")
        initialized = initialize_buffadvisor()
    
    ready = bot.advisor_instance is not None
    print(f"[DEBUG] Status check - ready: {ready}")
    
    response = {
        'ready': ready,
        'status': 'initialized' if ready else 'initializing',
        'mode': 'cpu' if os.environ.get('ORT_GENAI_DEVICE') == 'cpu' else 'gpu',
        'active_requests': len(request_threads)
    }
    
    return jsonify(response), 200


@app.route('/api/chat', methods=['POST', 'GET'])
def chat():
    global initialized, server_metrics, request_threads
    request_id = f"{time.time()}-{id(threading.current_thread())}"
    request_start_time = time.time()
    print(f"[DEBUG] Chat endpoint called - request_id: {request_id}, method: {request.method}")
    
    server_metrics["requests_processed"] += 1
    
    if not initialized and bot.advisor_instance is None:
        print("[DEBUG] Initializing advisor in chat endpoint")
        initialized = initialize_buffadvisor()
    
    if bot.advisor_instance is None:
        print("[ERROR] Advisor not initialized")
        server_metrics["errors"] += 1
        return jsonify({
            'error': 'BuffAdvisor not initialized',
            'message': 'The Python bot system is still initializing or no PDF file was found.',
            'status': 'not_ready'
        }), 503
    
    try:
        if request.method == 'POST':
            data = request.json
        else:
            data = request.args.to_dict()
        
        message = data.get('message', '')
        style = data.get('style', 'brief')
        new_session = data.get('new_session', 'false').lower() == 'true' if isinstance(data.get('new_session'), str) else bool(data.get('new_session', False))
        streaming = data.get('streaming', 'true').lower() == 'true' if isinstance(data.get('streaming'), str) else bool(data.get('streaming', True))
        
        if len(message) > 500:
            print(f"[DEBUG] Truncating long message from {len(message)} to 500 chars")
            message = message[:497] + "..."
            
        print(f"[DEBUG] Received message: '{message}'")
        print(f"[DEBUG] Style: {style}, New session: {new_session}, Streaming: {streaming}")
        
        if not message:
            print("[ERROR] No message provided")
            server_metrics["errors"] += 1
            return jsonify({'error': 'No message provided', 'status': 'error'}), 400
        
        if streaming:
            def generate():
                yield 'data: {"status": "started", "message": "Generation started"}\n\n'
                
                try:
                    start_time = time.time()
                    for text_chunk in bot.get_advice_stream(
                        new_session=new_session,
                        question=message,
                        style=style
                    ):
                        escaped_chunk = text_chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                        
                        yield f'data: {{"chunk": "{escaped_chunk}", "status": "generating"}}\n\n'
                        
                    processing_time = time.time() - start_time
                    server_metrics["total_response_time"] += processing_time
                    server_metrics["avg_response_time"] = server_metrics["total_response_time"] / server_metrics["requests_processed"]
                    
                    yield f'data: {{"status": "complete", "processing_time": {processing_time}}}\n\n'
                except Exception as e:
                    print(f"[ERROR] Streaming error: {str(e)}")
                    error_msg = str(e).replace('"', '\\"').replace('\\', '\\\\')
                    yield f'data: {{"status": "error", "message": "{error_msg}"}}\n\n'
            
            return Response(stream_with_context(generate()), 
                            mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 
                                    'X-Accel-Buffering': 'no'})
        
        MAX_PROCESSING_TIME = 120
        
        print(f"[DEBUG] Starting to process message: '{message}'")
        start_time = time.time()
        
        def process_with_timeout():
            try:
                return bot.get_advice(
                    new_session=new_session,
                    question=message,
                    style=style
                )
            except Exception as e:
                print(f"[ERROR] Exception during bot processing: {str(e)}")
                print(traceback.format_exc())
                return None
                
        result = [None]
        processing_thread = threading.Thread(target=lambda: result.__setitem__(0, process_with_timeout()))
        processing_thread.daemon = True
        
        request_threads[request_id] = {
            "thread": processing_thread,
            "start_time": time.time(),
            "message": message[:50] + "..." if len(message) > 50 else message
        }
        
        processing_thread.start()
        
        processing_thread.join(MAX_PROCESSING_TIME)
        
        if processing_thread.is_alive():
            print(f"[ERROR] Request timed out after {MAX_PROCESSING_TIME} seconds")
            server_metrics["timeouts"] += 1
            
            request_threads[request_id]["status"] = "timed_out"
            
            return jsonify({
                'error': 'Request timeout',
                'message': f'The request took too long to process (> {MAX_PROCESSING_TIME} seconds). The backend is still processing your request in the background.',
                'status': 'timeout',
                'request_id': request_id
            }), 504
            
        if request_id in request_threads:
            del request_threads[request_id]
            
        bot_response = result[0]
        
        if bot_response is None:
            server_metrics["errors"] += 1
            return jsonify({
                'error': 'Failed to generate response',
                'message': 'An error occurred during processing',
                'status': 'error'
            }), 500
            
        print(f"[DEBUG] Response generated successfully: '{bot_response[:100]}...'")
        
        processing_time = time.time() - start_time
        total_time = time.time() - request_start_time
        
        server_metrics["total_response_time"] += processing_time
        server_metrics["avg_response_time"] = server_metrics["total_response_time"] / server_metrics["requests_processed"]
        
        print(f"[DEBUG] Processing time: {processing_time:.2f}s, Total time: {total_time:.2f}s")
        
        return jsonify({
            'response': bot_response,
            'processing_time': processing_time,
            'source': 'bot.py',
            'model_used': 'DeepSeek R1',
            'mode': 'cpu' if os.environ.get('ORT_GENAI_DEVICE') == 'cpu' else 'gpu',
            'status': 'success'
        }), 200
    except Exception as e:
        error_message = str(e)
        print(f"[ERROR] Exception in chat endpoint: {error_message}")
        print(traceback.format_exc())
        server_metrics["errors"] += 1
        return jsonify({
            'error': 'Failed to process message with bot',
            'message': error_message,
            'status': 'error'
        }), 500


@app.route('/api/initialize', methods=['POST'])
def initialize():
    global initialized
    print("[DEBUG] Initialize endpoint called")
    data = request.json
    pdf_path = data.get('pdf_path')
    
    if not pdf_path:
        print("[ERROR] No PDF path provided")
        return jsonify({'error': 'No PDF path provided'}), 400
    
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        return jsonify({'error': f'PDF not found at path: {pdf_path}'}), 404
    
    try:
        print(f"[DEBUG] Initializing with PDF: {pdf_path}")
        def initialize_thread():
            try:
                bot.initialize_advisor(pdf_path)
                global initialized
                initialized = bot.advisor_instance is not None
            except Exception as e:
                print(f"[ERROR] Initialization thread error: {str(e)}")
                traceback.print_exc()
                
        thread = threading.Thread(target=initialize_thread)
        thread.daemon = True
        thread.start()
        
        print("[DEBUG] Initialization started in background thread")
        return jsonify({'status': 'Initialization started'}), 202
    except Exception as e:
        error_message = str(e)
        print(f"[ERROR] Initialization failed: {error_message}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Failed to initialize',
            'message': error_message
        }), 500


if __name__ == '__main__':
    print("[DEBUG] Server starting up")
    initialize_buffadvisor()
    
    port = int(os.environ.get('PORT', 5001))
    print(f"[DEBUG] Starting server on port {port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"[CRITICAL] Server failed to start: {str(e)}")
        traceback.print_exc()
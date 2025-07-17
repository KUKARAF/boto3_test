import boto3
import time
import json
import csv
import os
import sys
import select
import tty
import termios
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='model_benchmark.log'
)
logger = logging.getLogger('model_benchmark')

# Initialize Bedrock client with no retries and 30 second timeout
bedrock_runtime = boto3.client(
    'bedrock-runtime',
    config=boto3.session.Config(
        connect_timeout=30,
        read_timeout=30,
        retries={'max_attempts': 0}
    )
)

# Models to test
MODELS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "eu.amazon.nova-lite-v1:0",
    "amazon.titan-embed-text-v2:0",
    "cohere.rerank-v3-5:0"
]

# Load questions and bible text
def load_questions():
    with open('questions.txt', 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Load bible text to add to each request (approximately 6k tokens)
    with open('bible.txt', 'r') as f:
        bible_text = f.read(24000)  # ~6k tokens (4 chars per token)
        print(f"Loaded {len(bible_text)} characters (~{len(bible_text)//4} tokens) from bible.txt")
    
    # Add bible text to each question
    padded_questions = []
    for q in questions:
        padded_questions.append(f"{q}\n\nFor reference, here is some additional context:\n\n{bible_text}")
    
    return padded_questions

# Function to count tokens (simple approximation)
def count_tokens(text):
    # Rough approximation: 4 chars = 1 token
    return len(text) // 4

# Function to invoke model
def invoke_model(model_id, prompt):
    start_time = time.time()
    error = None
    response_text = ""
    
    print(f"Sending request to {model_id}...")
    print(f"Prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
    
    try:
        if "claude" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = response_body['content'][0]['text']
            
        elif "nova" in model_id:
            body = json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ]
                # Nova doesn't accept max_tokens parameter
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = response_body['output']
            
        elif "titan-embed" in model_id:
            body = json.dumps({
                "inputText": prompt
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = str(response_body['embedding'])[:100] + "..."  # Just show part of embedding
            
        elif "rerank" in model_id:
            body = json.dumps({
                "documents": [
                    {"text": "Paris is the capital of France"},
                    {"text": "London is the capital of England"},
                    {"text": "Berlin is the capital of Germany"}
                ],
                "query": prompt,
                "top_n": 1
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = str(response_body['results'])[:100] + "..."
    
    except Exception as e:
        error = str(e)
        logger.error(f"Error invoking {model_id}: {error}")
        print(f"ERROR: Failed to invoke {model_id}: {error}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "duration": duration,
        "error": error,
        "response": response_text,
        "input_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(response_text) if response_text else 0
    }

# Function to check if 'p' key was pressed
def is_p_pressed():
    # Store the terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Set the terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        # Check if there's input ready (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            return key == 'p'
    except Exception as e:
        print(f"Error checking for keypress: {e}")
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return False

# Main benchmark function
def run_benchmark():
    questions = load_questions()
    results = []
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"results/benchmark_{timestamp}.csv"
    
    print("\n=== Benchmark started ===")
    print("Press 'p' at any time to add more test iterations")
    
    # Write CSV header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id", "question_id", "duration", "input_tokens", 
            "output_tokens", "total_tokens", "tokens_per_minute", "success"
        ])
    
    # Run benchmark for each model and question
    for model_id in MODELS:
        logger.info(f"Testing model: {model_id}")
        model_results = []
        
        for i, question in enumerate(questions):
            logger.info(f"  Question {i+1}/{len(questions)}")
            
            result = invoke_model(model_id, question)
            success = result["error"] is None
            
            # Calculate total tokens and tokens per minute
            total_tokens = result["input_tokens"] + result["output_tokens"]
            tokens_per_minute = (total_tokens / result["duration"]) * 60 if result["duration"] > 0 else 0
            
            # Log result
            logger.info(f"    Duration: {result['duration']:.2f}s, Tokens/min: {tokens_per_minute:.2f}, Success: {success}")
            print(f"  Question {i+1}: Duration: {result['duration']:.2f}s, Tokens/min: {tokens_per_minute:.2f}, Success: {success}")
            print(f"    Tokens sent: {result['input_tokens']}, Tokens received: {result['output_tokens']}")
            
            # Error details are already printed in invoke_model function
            
            # Save to results list
            result_row = {
                "model_id": model_id,
                "question_id": i+1,
                "duration": result["duration"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": total_tokens,
                "tokens_per_minute": tokens_per_minute,
                "success": success
            }
            model_results.append(result_row)
            
            # Write to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_id, i+1, result["duration"], result["input_tokens"],
                    result["output_tokens"], total_tokens, tokens_per_minute, success
                ])
            
            # Small delay between requests
            time.sleep(1)
            
            # Check if 'p' was pressed to add more iterations
            if is_p_pressed():
                print("\n=== Adding more test iterations ===")
                print(f"Current model: {model_id}, Current question: {i+1}")
                print("How many more iterations of this question do you want to run?")
                try:
                    more_iterations = int(input("Enter number (default: 1): ") or "1")
                    for j in range(more_iterations):
                        print(f"\nRunning additional iteration {j+1}/{more_iterations}...")
                        
                        # Run the same question again
                        result = invoke_model(model_id, question)
                        success = result["error"] is None
                        
                        # Calculate total tokens and tokens per minute
                        total_tokens = result["input_tokens"] + result["output_tokens"]
                        tokens_per_minute = (total_tokens / result["duration"]) * 60 if result["duration"] > 0 else 0
                        
                        # Log result
                        logger.info(f"    Additional run {j+1}: Duration: {result['duration']:.2f}s, Tokens/min: {tokens_per_minute:.2f}, Success: {success}")
                        print(f"  Additional run {j+1}: Duration: {result['duration']:.2f}s, Tokens/min: {tokens_per_minute:.2f}, Success: {success}")
                        print(f"    Tokens sent: {result['input_tokens']}, Tokens received: {result['output_tokens']}")
                        
                        # Save to results list
                        result_row = {
                            "model_id": model_id,
                            "question_id": i+1,
                            "duration": result["duration"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "total_tokens": total_tokens,
                            "tokens_per_minute": tokens_per_minute,
                            "success": success
                        }
                        model_results.append(result_row)
                        
                        # Write to CSV
                        with open(csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                model_id, i+1, result["duration"], result["input_tokens"],
                                result["output_tokens"], total_tokens, tokens_per_minute, success
                            ])
                        
                        # Small delay between requests
                        time.sleep(1)
                    
                    print("Additional iterations completed.")
                except ValueError:
                    print("Invalid input. Continuing with regular benchmark.")
        
        results.extend(model_results)
    
    logger.info(f"Benchmark complete. Results saved to {csv_file}")
    return csv_file, results

if __name__ == "__main__":
    try:
        csv_file, results = run_benchmark()
        print(f"\nBenchmark complete. Results saved to {csv_file}")
        print("Run 'python generate_graph.py' to create visualization of the results.")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Partial results may have been saved.")
        print("Run 'python generate_graph.py' to create visualization of any saved results.")

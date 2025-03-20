import time
import datetime

def main():
    # Open (or create) a file in append mode
    with open("test_output.txt", "a") as f:
        counter = 0
        while True:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"[{current_time}] Iteration {counter}\n"
            f.write(message)
            
            # Force the buffer to flush so that the file is actually updated
            f.flush()
            
            counter += 1
            # Sleep for a short period so it doesn't overload the system
            time.sleep(2)

if __name__ == "__main__":
    main()
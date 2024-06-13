import socket

try:
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    s.bind(('0.0.0.0', 8080))

    # Listen for incoming connections
    s.listen(1)

    print("Listening for incoming connections...")
    conn, addr = s.accept()
    print(f"Connected to {addr}")

    while True:
        data = conn.recv(1024)
        if not data:
            print("Client disconnected. Exiting...")
            break
        print(f"Received: {data.decode('utf-8')}")

except KeyboardInterrupt:
    print("\nServer is shutting down...")
    if 'conn' in locals():
        conn.close()
    s.close()

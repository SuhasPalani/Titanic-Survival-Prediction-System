# scripts/start_kafka.sh
#!/bin/bash

echo "Starting Kafka services..."

# Start Docker Compose services
docker-compose up -d

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 30

# Create topic if it doesn't exist
docker exec -it $(docker-compose ps -q kafka) kafka-topics --create --topic titanic-predictions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --if-not-exists

echo "Kafka is ready!"
echo "To stop Kafka: docker-compose down"
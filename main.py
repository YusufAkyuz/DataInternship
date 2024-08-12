import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

# Log file path
log_file = '/private/var/log/apache2/access_log'

# Empty list for required data
cleaned_logs = []

# Read data and push to list
with open(log_file, 'r') as f:
    logs = f.readlines()

cleaned_logs = [(parts[0], parts[3].strip('[') + ' ' + parts[4].strip(']'), parts[6])
                for log in logs
                for parts in [log.split()]
                if len(parts) >= 7]

# Save the cleaned data to a CSV file
csv_file = 'cleaned_logs.csv'

with open(csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['IP Address', 'Timestamp', 'URL'])
    csv_writer.writerows(cleaned_logs)

print(f'Cleaned logs have been saved to {csv_file}')

# Load cleaned logs into DataFrame
df = pd.DataFrame(cleaned_logs, columns=['IP Address', 'Timestamp', 'URL'])

# Vectorize the URL data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))  # Adjust parameters
X = vectorizer.fit_transform(df['URL'])

# Convert vectors to numpy array
vectors = X.toarray().astype(np.float32)

# Print vector dimension
print(f'Vector dimension: {vectors.shape[1]}')

# Check for non-zero vectors
if np.count_nonzero(vectors) == 0:
    raise ValueError("All vectors are zero vectors. Check the input data and vectorizer parameters.")

# Initialize Pinecone
api_key = 'c6c168dd-b1de-423d-b1ed-48a98cbdc0ad'
pc = Pinecone(api_key=api_key)

# Create Pinecone index if it does not exist
index_name = 'logdata'  # Ensure the index name is lowercase and valid

# Check if index exists
indexes = pc.list_indexes().names()
if index_name in indexes:
    # If the index exists, delete it
    pc.delete_index(index_name)

# Create new index with correct dimension
dimension = vectors.shape[1]
try:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Updated region based on the free plan's available regions
        )
    )
except Exception as e:
    print(f"Error creating index: {e}")

index = pc.Index(index_name)

index_info = pc.describe_index(index_name)
print(index_info)

# Upsert vectors to Pinecone index
ids = [str(i) for i in range(len(vectors))]

# Ensure non-zero vectors before upserting
non_zero_vectors = [(id_, vec.tolist()) for id_, vec in zip(ids, vectors) if np.any(vec != 0)]
index.upsert(vectors=non_zero_vectors)

# Query generation and vectorization (modify as needed)
query = '::1,12/Aug/2024:12:20:49 +0300,/key=ali'
query_vector = vectorizer.transform([query]).toarray().astype(np.float32).tolist()[0]
print(vectors[0])
try:
    result = index.query(queries = vectors[0], top_k=2)
    print(result)
except Exception as e:
    print(f"Error querying index: {e}")

# Base image with Node.js + Python
FROM node:18-bullseye

COPY requirements.txt .

# Install Python and DVC
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    pip3 install dvc &&  pip install -r requirements.txt

# # Install Python dependencies
# RUN pip install --upgrade pip \
#     && pip install -r requirements.txt

# Set working directory
WORKDIR /app

# Copy root package.json (if exists)
COPY package*.json ./

# Install root dependencies
RUN npm install

# Copy client and server code
COPY client ./client
COPY server ./server

# Install client/server dependencies
RUN cd client && npm install
RUN cd server && npm install

# Copy DVC config
COPY .dvc ./.dvc

# Initialize DVC (if not already initialized)
RUN if [ ! -d .dvc ]; then dvc init --no-scm; fi

# Set up local remote (adjust path as needed)
RUN dvc remote add -d mylocal /dvc-storage

# Pull data during build (optional)
# RUN dvc pull

EXPOSE 3000 5000
CMD ["npm", "start"]

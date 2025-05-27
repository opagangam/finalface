# Use Conda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yaml .

# Create Conda environment
RUN conda env update -f environment.yaml && conda clean -afy

# Set PATH so conda env commands work in CMD/RUN
ENV PATH /opt/conda/envs/faceapp/bin:$PATH

# Copy rest of the app
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run your app
CMD ["conda", "run", "--no-capture-output", "-n", "faceapp", "python", "app.py"]

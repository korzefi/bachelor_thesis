FROM covid-base:ubuntu

USER root

# Set Python path for module imports
ENV PYTHONPATH="${PYTHONPATH}:/bachelor_thesis"

# Set working directory
WORKDIR /bachelor_thesis

# Copy application code (dependencies already installed in base image)
COPY . .

# Set proper permissions for main script
RUN chmod +x main.py

# Health check to verify the application is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import scripts.config; print('OK')" || exit 1

# Default command shows help - can be overridden with specific pipeline steps
# Examples:
# docker run app --prepare
# docker run app --cluster  
# docker run app --train
CMD ["python3", "main.py", "--help"]

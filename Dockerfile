# Add a line here to specify the docker image to inherit from.
FROM tensorflow/tensorflow:latest

USER root

ARG WORK_DIR="/app"
WORKDIR $WORK_DIR

# Add lines here to copy over your src folder and
# any other files you need in the image (like the saved model).
COPY requirements.txt $WORK_DIR/requirements.txt

# Add a line here to update the conda environment using the conda.yml.
# Remember to specify that the environment to update is 'polyaxon'.
RUN python3 -m pip install -r requirements.txt

COPY src/ $WORK_DIR/src
COPY model/ $WORK_DIR/model

EXPOSE 8000

# Add a line here to run your app
ENTRYPOINT ["python"]
CMD ["./src/app.py"]




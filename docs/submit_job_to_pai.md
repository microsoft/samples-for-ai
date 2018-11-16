# Submit a job to Open Platform for AI (short for OpenPAI)

[**Microsoft OpenPAI**](https://github.com/Microsoft/pai) is an open source platform that provides complete AI model training and resource management capabilities. Most samples can submit to OpenPAI cluster. For each project, a json file is provided as an example to show how to configure the submit information. User can just use the example json file or set your specified configuration.

## Submit from Microsoft Visual Studio Tools for AI

### 1. [Install Microsoft Visual Studio Tools for AI (short for VS Tools for AI)](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/installation.md)

### 2. [Prepare environment](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md)

### 3. Add OpenPAI cluster and submit job

#### (1) [Add OpenPAI cluster](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/pai.md#Add%20a%20PAI%20cluster)

#### (2) Submit Job to OpenPAI
   
   - Right-Click project name -> "Submit Job...".
   - In the pop-up dialog window, select your OpenPAI cluster.
   - Write your own configuration or "Import" json file.
   - If you want use example json file as configuration: Click "Import..." button, select one json file
   - Click "Submit".
        
   Please visit [Job Submission to a PAI cluster](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/pai.md#Job%20Submission%20to%20a%20PAI%20cluster) for details.


## Submit from Microsoft Visual Studio Code Tools for AI

### 1. [Install Microsoft Visual Studio Code Tools for AI (short for VSCode Tools for AI)](https://github.com/Microsoft/vscode-tools-for-ai/blob/master/docs/installation.md)

### 2. [Prepare environment](https://github.com/Microsoft/vscode-tools-for-ai/blob/master/docs/prepare-localmachine.md)

### 3. [Add OpenPAI cluster and submit job](https://github.com/Microsoft/vscode-tools-for-ai/blob/master/docs/quickstart-05-pai.md)

## Submit from OpenPAI web portal

### 1. Upload code and data to HDFS storage on OpenPAI cluster.

   **Method 1:** Use hdfs command to upload local file and data to HDFS storage. See [here](https://hadoop.apache.org/docs/r2.7.1/hadoop-project-dist/hadoop-hdfs/HDFSCommands.html#dfs) to look up HDFS command.

   **Method 2:** Use _Tools for AI_ to upload code and data via GUI. See [here](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/pai.md) for details.

   In these samples, we uploaded code to $PAI_DEFAULT_FS_URI/tutorial/**_sample_name_**/code directory, and dataset to $PAI_DEFAULT_FS_URI/tutorial/**_sample_name_**/data directory if needed.

   **NOTICE**: Most samples can download data from website automatically in the code, so you don't need to upload data manually. For details please view README.md file in each framework folder.


### 2. Submit job to webportal

   View [submit a job from web portal](https://github.com/Microsoft/pai/blob/master/docs/submit_from_webportal.md) to get the steps.

   We provide json filse as submission configuration for some samples. You can import that file directly. Each json file is named with the format: **framework_name.sample_name.json**, you can find them in the project folder.

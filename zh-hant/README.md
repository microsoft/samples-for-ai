# 引言

[English](/README.md) [简体中文](/zh-hans/README.md)

透過 Visual Studio 專案格式的範例，讓使用者學習如何使用 [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai) 來開發深度學習專案，而每個解決方案包含了一或多個範例專案。

解決方案則是以它們所使用不同的深度學習框架來區分：
- CNTK (同時包括 BrainScript 以及 Python 程式語言版本)
- Tensorflow
- Caffe2
- Keras
- MXNet
- Chainer
- Theano

# 參與貢獻

本專案歡迎使用者參與貢獻或是提出建言，大部份貢獻的內容，您必須同意一份參與者授權協議（Contributor License Agreement, CLA），協議中聲明您有權且實際授權我們使用您的貢獻內容。如需瞭解更多細節，請參考
https://cla.microsoft.com 網站說明。

當您提交一個 pull request (PR) 時，CLA-bot 將會自動確認您是否需要提交一份 CLA 而在您的 PR 內容中做適當地標示（例如，使用標籤或留言），這只要簡單地遵照 bot 所提示的步驟進行即可，而在所有使用我們參與者授權協議的存儲庫中，這個動作您只需要做一次。

這個專案遵守 [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) 規範。想要瞭解更多訊息，請參考 [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
或是聯絡 [opencode@microsoft.com](mailto:opencode@microsoft.com) 來詢問相關問題或留言。

# 開始

## 執行這些範例程式前的準備工作
- 安裝 [Microsoft Visual Studio](https://www.visualstudio.com/) 2017 or 2015 版本。
- 安裝 [Microsoft Visual Studio Tools for AI](https://github.com/Microsoft/vs-tools-for-ai) 這個 Visual Studio 的擴充套件。
- 預先下載資料
    - 在 CNTK BrianScript MNIST 專案中，於 "input" 目錄下，執行 "python install_mnist.py" 指令來下載資料

## 準備開發環境
在您的電腦或遠端機器上開始訓練度學習模型之前，請確認您已經安裝了相關的深度學習軟體，包括為了 NVIDIA GPU（如果有的話）最新的驅動程式以及相關的函式庫，另外，您也必須安裝 Python 及其相關的函式庫如 NumPy、SciPy、Visual Studio 中的 Python 開發支援、以及相關的深度學習框架如 Microsoft Cognitive Toolkit (CNTK)、TensorFlow、Caffe2、MXNet、Keras、Theano、PyTorch 以及 Chainer。

請參考[這裡](https://github.com/Microsoft/vs-tools-for-ai/blob/master/docs/prepare-localmachine.md)取得詳細的步驟來安裝這些環境。

## 使用一鍵安裝工具來安裝深度學習框架

目前，這個安裝工具可支援 Windows、macOS 以及 Linux：

- 安裝最新的 NVIDIA GPU 驅動程式、CUDA 9.0 以及 cuDNN 函式庫 7.0。
- 安裝最新的 **Python 3.5 或 3.6**，目前並不支援其它 Python 版本。
- 在命令列模式（或終端機）下執行下列的指令：
   > [!注意]
   >
   > - 如果您的 Python 套件安裝在系統目錄（例如 Visual Studio 2017 附帶的，或是在 Linux 中內建的），您必須以系統管理者的權限來執行安裝程式。
   > - 傳入 "**--user**" 參數以便讓安裝程式進行 Python 使用者層級的安裝，也就是將函式庫安裝在像是 ~/.local/ 或是 Windows 中的 %APPDATA%\Python 目錄下。
   > 安裝程式會偵測系統中是否有 NVIDIA GPU 顯示卡，並且以 CUDA 9.0 作為預設環境來設定安裝的軟體，您可以使用 "**--cuda80**" 參數來強制設定使用 CUDA 8.0 來做設定。
   
   ```bash
   git clone https://github.com/Microsoft/samples-for-ai.git
   cd samples-for-ai
   cd installer
   - Windows:
       python.exe install.py
   - Non-Windows:
       python3 install.py
   ```

## 在本地電腦執行範例程式

- CNTK BrainScript 專案
    - 將您要執行的專案設定成 "啟始專案"
    - 將您要執行的指令碼設定成 "啟動檔案"
    - 右鍵點擊 "執行 CNTK Brian Script"

- Python 專案
    - 設定 "啟動檔案"
    - 右鍵點擊要執行的 Python script，在選單中選擇 "啟動但不偵錯" 或是 "啟動偵錯"。


# 授權聲明

範例程式中的 script 檔案都是來自於不同框架官方的 GitHub 存儲庫中，它們分別使用不同的授權。

CNTK 中的 script 使用 [MIT 授權](https://en.wikipedia.org/wiki/MIT_License)。

Tensorflow 中的 script 使用 [Apache 2.0 授權](https://en.wikipedia.org/wiki/Apache_License#Version_2.0)，而這裡所收錄的程式並未被修改。

針對 Caffe2 中的 script，在不同的版本中使用了不同的授權。目前主要的分支採用 Apache 2.0 授權，但是 0.7 以及 0.8.1 的版本是使用 [BSD 2-Clause 授權](https://github.com/caffe2/caffe2/tree/v0.8.1)，在這裡的程式碼都是基於 caffe2 GitHub 中的 0.7 以及 0.8.1 的版本，所以是使用 BSD 2-Clause 授權。

Keras 中的程式使用 [MIT 授權](https://github.com/fchollet/keras/blob/master/LICENSE)。

Theano 中的程式使用 [BSD 授權](https://en.wikipedia.org/wiki/BSD_licenses)。

MXNet 中的程式使用 [Apache 2.0 授權](https://en.wikipedia.org/wiki/Apache_License#Version_2.0)，而這裡所收錄的程式並未被修改。

Chainer 中的程式使用 [MIT 授權](https://github.com/chainer/chainer/blob/master/LICENSE)。
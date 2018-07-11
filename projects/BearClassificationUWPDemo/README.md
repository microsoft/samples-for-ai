# Introduction

This is a simple UWP application demonstrates how to use Custom Vision for model training and exporting and Windows ML for creating model inference library.  

[Custom Vision](http://customvision.ai)

[Windows ML](https://docs.microsoft.com/en-us/windows/uwp/machine-learning/)

# Prerequisites

1. Required Win10 version 1803.
2. Open Visual Studio Installer -> Modify -> Workloads -> Windows -> install Universal Windows Platform development
3. Open Visual Studio Installer -> Modify -> Individual components -> SDKs, libraries, and frameworks -> install Windows 10 SDK (10.0.17134.0)

# Getting Started

1. Open the **BearClassificationUWP.App.sln** solution, and run it.
2. Paste the URL of a picture into TextBox, make sure it starts with http(s). Or you can browse local file and choose one.
3. Click Recognize to get result.

# Code Overview

There are two .cs files that contain the key code : **/ViewModel/ResultViewModel.cs** and **BearClassification.cs**.
- **/ViewModel/ResultViewModel.cs** contains code of model loading, picture evaluation and description generation.
- **BearClassification.cs** was generated automatically, it contains three classes which are wrappers of the model, its input and output. Garbled was replaced to "BearClassification" for better reading experience.

## ResultViewModel.cs

This file contains all functional code.

1. Load the model

```c#
private async void LoadModel()
{
	StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/BearClassification.onnx"));
	model = await BearClassificationModel.CreateBearClassificationModel(modelFile);
}
```

2. Get picture from URL
In function : EvaluateAsync

```c#
var response = await new HttpClient().GetAsync(BearUrl);
var stream = await response.Content.ReadAsStreamAsync();
BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream.AsRandomAccessStream());
VideoFrame imageFrame = VideoFrame.CreateWithSoftwareBitmap(await decoder.GetSoftwareBitmapAsync());
```


3. Evaluate the input,
In function : EvaluateAsync

```c#
var result = await model.EvaluateAsync(new BearClassificationModelInput() { data = imageFrame });
var resultDescend = result.loss.OrderByDescending(p => p.Value).ToDictionary(p => p.Key, o => o.Value);
```

You can read this tutorial to know how to [integrate a model into your app with Windows ML](https://docs.microsoft.com/en-US/windows/uwp/machine-learning/integrate-model).

Chinese version: [使用 Windows ML 将模型集成到你的应用中](https://docs.microsoft.com/zh-cn/windows/uwp/machine-learning/integrate-model)

# How to integrate a model

If you have .onnx model files. You can create a Model inference Library using these files.

Steps,

1. Add ONNX model to the project's **Assets** folder, then Visual Studio will automatically generate the wrapper classes in a new file.
2. Change your .onnx file's properties: `Advanced->Buid Action` to **Content**
3. Change the .onnx file path in function **LoadModel**
```c#
StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/changetoyourfile.onnx"));
```
4. Change items in **/Strings/en-US/Resources.resw** and **/Strings/zh-CN/Resources.resw** when you need to.

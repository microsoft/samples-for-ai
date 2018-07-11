using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MNISTModelLibrary;// reference MNISTModelLibrary

namespace MNIST.App
{
    public partial class MainWindow : Form
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private const int MnistImageSize = 28;// the input image size of MnistModel

        private Mnist model;// MNIST model
        private Graphics graphics;
        private Point startPoint;// coordinate of the start point of the line to be draw
        
        // Initialize the window, setup all things
        private void Form1_Load(object sender, EventArgs e)
        {
            model = new Mnist();
            writeArea.Image = new Bitmap(writeArea.Width, writeArea.Height);
            graphics = Graphics.FromImage(writeArea.Image);
            graphics.Clear(Color.White);
        }

        // Erase the content of the image and the text of label1 after erase button been clicked
        private void clean_click(object sender, EventArgs e)
        {
            graphics.Clear(Color.White);
            writeArea.Invalidate();
            outputText.Text = "";
        }

        // Record the start coordinate of the line  after left button been pressed
        private void writeArea_MouseDown(object sender, MouseEventArgs e)
        {
            startPoint = (e.Button == MouseButtons.Left) ? e.Location : startPoint;
        }

        // Draw the line according to the start coordinate and the current coordinate
        // Update the start coordinate of the next line when the mouse is moving
        private void writeArea_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                Pen penStyle = new Pen(Color.Black, 40) { StartCap = LineCap.Round, EndCap = LineCap.Round };
                graphics.DrawLine(penStyle, startPoint, e.Location);
                writeArea.Invalidate();
                startPoint = e.Location;
            }
        }

        //Normalize the picture and send it for inference after left button been released
        private void writeArea_MouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                /*
                  * Overview : normalize the picture and send it to the model
                  * The model can infer multiple images at once and return multiple results
                  * The input of mnist model is a list of "image" 
                  * Each "image" is a float list which records every pixel's grayscale value of a 28*28 image
                  * 
                  * In this demo program, we only send one image to infer at once
                  * So, before the inference, we need to resize the image, covert it to gray and get the grayscale value of each pixel    
                  * Then we need to put the image into a new list and send the list to the model
                  * When displaying the result, we need to get the predicted digit of the highest probability of the first image
                */

                // 1. begin to normalize data
                Bitmap clonedBmp = new Bitmap(MnistImageSize, MnistImageSize);
                Graphics gNormalized = Graphics.FromImage(clonedBmp);

                // a. normalize the size to 28*28 as the training input
                gNormalized.DrawImage(writeArea.Image, 0, 0, MnistImageSize, MnistImageSize);

                // b. normalize the data structure to a float list as training input
                var image = new List<float>(MnistImageSize * MnistImageSize);
                for (var x = 0; x < MnistImageSize; x++)
                {
                    for (var y = 0; y < MnistImageSize; y++)
                    {
                        var color = clonedBmp.GetPixel(y, x);
                        // c. normalize value
                        //      RGB to gray
                        //      Range to [-0.5,0.5]
                        image.Add((float)(0.5 - (color.R + color.G + color.B) / (3.0 * 255)));
                    }
                }
                // end to normalize data

                // 2. inferencing
                //      a. put the image into a new List 
                //      b. get the predicted digit of the highest probability of the first image
                outputText.Text = (model.Infer(new List<IEnumerable<float>> { image })).First().First().ToString();
            }
        }
    }
}

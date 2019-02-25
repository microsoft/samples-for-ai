using PictureStyleTransfer.ViewModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Data;

namespace PictureStyleTransfer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        public MainWindow()
        {
            InitializeComponent();
            this.Closing += HanldeClose;
        }

        public event PropertyChangedEventHandler PropertyChanged;
        
        private void UpateSource(object sender, DataTransferEventArgs e)
        {
        }

        private void HanldeClose(object sender, CancelEventArgs e)
        {
            this.IsEnabled = false;
            MainViewModel model = this.DataContext as MainViewModel;
            if (model != null)
            {
                model.Cleanup();
            }
            e.Cancel = false;
        }
    }
}

using Windows.UI.Xaml.Controls;
using CustomVisionUWP.App.ViewModel;

namespace CustomVisionUWP.App
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            this.InitializeComponent();
            //Binding data and command
            this.DataContext = new ResultViewModel();
        }
    }
}

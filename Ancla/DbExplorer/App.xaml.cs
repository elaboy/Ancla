using Database;

namespace DbExplorer
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            var dbContext = new PsmContext();

            MainPage = new DbExplorer.MainPage(dbContext);
        }
    }
}

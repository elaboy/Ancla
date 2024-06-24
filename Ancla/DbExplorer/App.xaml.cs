using Ancla;
using Database;

namespace DbExplorer
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            var dbContext = new DbContext();

            MainPage = new DbExplorer.MainPage(dbContext);
        }
    }
}

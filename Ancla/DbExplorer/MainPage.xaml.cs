using AnchorLib;
using Database;

namespace DbExplorer;
public partial class MainPage : ContentPage
{
    int count = 0;

    private readonly PsmContext _context;
    public MainPage(PsmContext psmContext)
    {
        _context = psmContext;

        InitializeComponent();

        PSMs.ItemsSource = _context.PSMs.ToList();
    }

    public void OnModificationClick(object? sender, EventArgs e)
    {
        // Change the button color to light red when clicked
        Button button = (Button)sender;
        if (button.BackgroundColor.Equals(Colors.LightBlue))
        {
            button.BackgroundColor = Colors.Coral;
        }
        else
        {
            button.BackgroundColor = Colors.LightBlue;
        }
    }

    private void Button_OnClicked(object? sender, EventArgs e)
    {
        OnModificationClick(sender, e);

        Button button = (Button)sender;
        if (button.BackgroundColor.Equals(Colors.Coral))
        {
            List<PSM> psms = _context.PSMs.ToList();
            List<PSM> phosphoPsms = new List<PSM>();

            foreach (var psm in psms)
            {
                if (psm.FullSequence.Contains("Phospho"))
                {
                    phosphoPsms.Add(psm);
                }
            }

            PSMs.ItemsSource = phosphoPsms;
        }
        else
        {
            PSMs.ItemsSource = _context.PSMs.ToList();
        }
    }
}


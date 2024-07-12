namespace AnchorCommandLine.RetentionTime;
public interface IRetentionTimeHarmonizer
{
    public string FileName { get; set; }
    public double RetentionTime { get; set; }
    public string Identifier { get; }
}

using AnchorLib;
using Readers;
namespace Database;
public class PsmService
{
    public static List<PSM> GetPsms(List<string> paths)
    {
        List<PSM> psms = new List<PSM>();

        foreach (var path in paths)
        {
            var psmtsv = SpectrumMatchTsvReader.ReadPsmTsv(path,
                out List<string> warnings);

            foreach (var psm in psmtsv)
            {
                psms.Add(new PSM(psm));
            }
        }

        return psms;
    }
}

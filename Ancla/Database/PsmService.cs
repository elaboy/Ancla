using AnchorLib;
using Database;
using Microsoft.EntityFrameworkCore;
using Readers;
namespace Database;
public class PsmService
{
    private readonly PsmContext _context;

    public PsmService(PsmContext context)
    {
        _context = context;
    }

    public List<PSM> GetPsms(List<string> paths)
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

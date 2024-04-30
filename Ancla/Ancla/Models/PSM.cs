using Readers;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Proteomics.PSM;

namespace Ancla.Models
{
    public class PSM
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }

        public string FileName { get; set; }
        public int ScanNumber { get; set; }
        public double ScanRetentionTime { get; set; }
        public double TotalIonCurrent { get; set; }
        public int PrecursorCharge { get; set; }
        public double PrecursorMZ { get; set; }
        public double PrecursorMass { get; set; }
        public double Score { get; set; }
        public string BaseSequence { get; set; }
        public string FullSequence { get; set; }
        public string AmbiguityLevel { get; set; }
        //public string Modifications { get; set; }
        public string PeptideMonoisotopicMass { get; set; }
        public string ProteinAccession { get; set; }
        public string ProteinName { get; set; }
        public string GeneName { get; set; }
        public string OrganismName { get; set; }
        public string StartAndEndResidueInProtein { get; set; }
        //public string MatchedIonSeries { get; set; }
        //public string MatchedIonMassToChargeRatios { get; set; }
        //public string MatchedIonIntensities { get; set; }
        //public int MatchedIonCount { get; set; }
        public double PEP { get; set; }
        public double QValue { get; set; }
        public double PEPQvalue { get; set; }

        public PSM()
        {
        }
    }
}

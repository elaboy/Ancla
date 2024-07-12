using System.Data;
using System.Globalization;
using AnchorCommandLine;
using AnchorCommandLine.RetentionTime;
using CsvHelper;

namespace TestCMD
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void TestA549CalibrationConstructor()
        {
            var calibrator = new Calibrator(@"D:\MannPeptideResults\A549_AllPSMs.psmtsv");
        }

        [Test]
        public void TestA549CalibrationCalibration()
        {
            var calibrator = new Calibrator(@"D:\MannPeptideResults\A549_AllPSMs.psmtsv");
            calibrator.Calibrate();
        }

        [Test]
        public void TestHarmonizerWithA549()
        {
            List<string> warnings = new();
            var harmonizer = new Harmonizer(@"D:\MannPeptideResults\A549_AllPSMs.psmtsv", out warnings);
            harmonizer.Calibrate();

            Assert.Pass();
        }
    }
}
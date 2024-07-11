using System.Data;
using System.Globalization;
using AnchorCommandLine;
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
    }
}
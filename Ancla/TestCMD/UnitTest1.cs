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
            
            harmonizer.Calibrate(1, 2);

            Assert.Pass();


        }


        [Test]
        public void F()
        {
            var set1 = Enumerable.Range(0, 100).Select(p => (double)p).ToList();
            var set2 = Enumerable.Repeat(0.0, 100).ToList();
            FindValueInSet1ThatSumsTo40(set1, set2);


        }

        public int FindValueInSet1ThatSumsTo40(List<double> set1, List<double> set2)
        {
            int numberTOReturn = -1;
            foreach (var result in SumTwoSets(set1, set2))
            {
                if (Math.Abs(result - 40.0) > 0.0001)
                {
                    numberTOReturn = (int)result;
                    break;

                }
            }

            return numberTOReturn;
        }

        public IEnumerable<double> SumTwoSets(List<double> set1, List<double> set2)
        {
            for (int i = 0; i < set1.Count; i++)
            {
                yield return set1[i] + set2[i];
            }
        }

    }
}
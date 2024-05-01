using AnchorLib;
using Database;
using TestContext = NUnit.Framework.TestContext;

namespace Tests.AnchorLibTests;
public class TestPsm
{
    [Test]
    public void TestConstructor()
    {
        var psmFilePath = new List<string>()
        {
            Path.Combine(TestContext.CurrentContext.TestDirectory, "ExcelEditedPeptide.psmtsv")
        };
        var psm = PsmService.GetPsms(psmFilePath);

        // assert that there is only one PSM object in the psm variable
        Assert.That(psm.Count == 1);
    }
}


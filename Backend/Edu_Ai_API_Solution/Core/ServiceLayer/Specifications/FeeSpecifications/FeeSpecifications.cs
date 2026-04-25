using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.FeeSpecifications
{
    public class FeeSpecifications : BaseSpecification<Fee, int>
    {
        public FeeSpecifications(int academicYearId) : base(p => p.AcademicYearId == academicYearId)
        {
            AddInclude(p => p.AcademicYear);
        }

        public FeeSpecifications(int academicYearId, FeeType feeType)
    : base(p => p.AcademicYearId == academicYearId &&
                p.Name == feeType)
        {
            AddInclude(p => p.AcademicYear);
        }

    }
}

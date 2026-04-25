using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.AcademicYearSpecifications
{
    public class AcademicYearSpecifications : BaseSpecification<AcademicYear, int>
    {
        public AcademicYearSpecifications() :base(null)
        {
            AddInclude(p => p.Fees);
        }
        public AcademicYearSpecifications(int academicYearId) : base(p => p.Id == academicYearId)
        {
            AddInclude(p => p.Fees);
        }

       
    }
}

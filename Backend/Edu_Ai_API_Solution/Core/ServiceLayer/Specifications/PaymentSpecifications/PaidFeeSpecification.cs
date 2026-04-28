using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications.PaymentSpecifications
{
    public class PaidFeeSpecification : BaseSpecification<Payment, int>
    {
        public PaidFeeSpecification(string studentId, int academicYearId)
            : base(p =>
                p.StudentId == studentId &&
                p.AcademicYearId == academicYearId &&

                p.Status == PaymentStatus.Paid
            )
        {
        }
    }
}

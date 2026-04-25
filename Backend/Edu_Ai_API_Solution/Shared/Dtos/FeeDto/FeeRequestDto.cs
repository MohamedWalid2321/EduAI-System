using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.FeeDto
{
    public class FeeRequestDto
    {
        public int AcademicYearId { get; set; }
        public decimal Amount { get; set; }

        public string name { get; set; }
    }
}

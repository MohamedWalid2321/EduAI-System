using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.FeeDto
{
    public class FeeResponseDto
    {
        public int academicYearId { get; set; }
        public decimal amount { get; set; }
        public string name { get; set; }

    }
}

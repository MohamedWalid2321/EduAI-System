using Shared.Dtos.FeeDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Shared.Dtos.AcademicYearDto
{
    public class AcademicYearDto
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public ICollection<FeeResponseDto> fees { get; set; }
    }
}

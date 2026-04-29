using DomainLayer.Enums;
using Shared.Dtos.AcademicYearDto;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceAbstractionLayer
{
    public interface IAcademicYearService
    {
        Task<List<AcademicYearDto>> GetAllAsync();
        Task<AcademicYearDto> GetByIdAsync(int Id);
        Task<AcademicYearDto> CreateAsync(string name);
    }
}

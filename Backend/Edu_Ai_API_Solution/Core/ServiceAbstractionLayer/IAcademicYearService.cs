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
        Task<List<AcademicYearDto>> GetAllAsync(CancellationToken cancellationToken = default);
        Task<AcademicYearDto> GetByIdAsync(int Id, CancellationToken cancellationToken = default);
        Task<AcademicYearDto> CreateAsync(string name, CancellationToken cancellationToken = default);
    }
}

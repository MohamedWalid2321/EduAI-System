using DomainLayer.Contracts;
using DomainLayer.Models;
using Mapster;
using Microsoft.VisualBasic;
using ServiceAbstractionLayer;
using ServiceLayer.Specifications.CourseSpecification;
using ServiceLayer.Specifications.DepartmentSpecification;
using Shared.Dtos;
using Shared.Dtos.CourseDto.Request;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
	public class DepartmentService(IUnitOfWork unitOfWork) : IDepartmentService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;

		public async Task<DepartmentDto> CreateOrUpdateDepartmentAsync(DepartmentDto createDepartmentDto)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var DepartmentEntity = createDepartmentDto.Adapt<Department>();

			if (createDepartmentDto.Id > 0)
			{
				//Update
				var FoundedDepartmentEntity = await departmentRepository.GetByIdAsync(createDepartmentDto.Id);
				if (FoundedDepartmentEntity is null)
				{
					throw new Exception($"this department with id : {createDepartmentDto.Id} is not found");
				}
				departmentRepository.Update(DepartmentEntity);
			}
			else {
				//create
				await departmentRepository.AddAsync(DepartmentEntity);
			}
			await _unitOfWork.SaveChangesAsync();
			return DepartmentEntity.Adapt<DepartmentDto>();
		}

		public async Task DeleteDepartmentAsync(int departmentId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentId);
			if (DepartmentEntity is null)
			{
				throw new Exception($"this department with id : {departmentId} is not found");
			}
			departmentRepository.Delete(DepartmentEntity!);
			await _unitOfWork.SaveChangesAsync();	
		}

		

		public async Task<DepartmentDto?> GetDepartmentByIdAsync(int departmentId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin);
			return DepartmentEntity.Adapt<DepartmentDto?>();
		}

		public async Task<IEnumerable<DepartmentDto>> GetAllDepartmentsAsync()
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification();
			var departments = await departmentRepository.GetAllAsync(departmentSpecificatioin);
			var departmentDtos = departments.Adapt<IEnumerable<DepartmentDto>>();
			return departmentDtos;
		}
		public async Task<IEnumerable<CourseRequestDto>> GetAllCourseBydepartmentIdAsync(int departmentId)
		{
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseSpecification = new CourseByDepartmentSpecification(departmentId);
			var courses = await CourseRepository.GetAllAsync(courseSpecification);
			return courses.Adapt<IEnumerable<CourseRequestDto>>();
		}

		public async Task<DepartmentDto> AssignCourseToDepartmentAsync(int departmentId, int CourseId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await  departmentRepository.GetByIdAsync(departmentSpecificatioin);
			if (DepartmentEntity is null)
			{
				throw new Exception($"this department with id : {departmentId} is not found");
			}
			var CourseRepository = _unitOfWork.GetRepository<Course, int>();
			var courseEntity = await CourseRepository.GetByIdAsync(CourseId);
			if (courseEntity is null)
			{
				throw new Exception($"this course with id : {CourseId} is not found");
			}
			if (DepartmentEntity.courses.Any(c => c.Id == CourseId))
			{
				throw new Exception($"this course with id : {CourseId} is already assigned to department with id : {departmentId}");
			}
			DepartmentEntity.courses.Add(courseEntity);
			departmentRepository.Update(DepartmentEntity);
			await _unitOfWork.SaveChangesAsync();
			return DepartmentEntity.Adapt<DepartmentDto>();
		}

		public async Task<DepartmentDto> RemoveCourseFromDepartmentAsync(int departmentId, int CourseId)
		{
			var departmentRepository = _unitOfWork.GetRepository<Department, int>();
			var departmentSpecificatioin = new DepartmentSpecification(departmentId);
			var DepartmentEntity = await departmentRepository.GetByIdAsync(departmentSpecificatioin);
			if (DepartmentEntity is null)
			{
				throw new Exception($"this department with id : {departmentId} is not found");
			}
			if (DepartmentEntity.courses is null) {
				throw new Exception($"this department with id : {departmentId} has no courses assigned");
			}
			if (!DepartmentEntity.courses.Any(c => c.Id == CourseId))
			{
				throw new Exception($"this course with id : {CourseId} is not assigned to department with id : {departmentId}");
			}
			var courseEntity = DepartmentEntity.courses.FirstOrDefault(c => c.Id == CourseId);
			DepartmentEntity.courses.Remove(courseEntity!);
			departmentRepository.Update(DepartmentEntity);
			await _unitOfWork.SaveChangesAsync();
			return DepartmentEntity.Adapt<DepartmentDto>();
		}
	}
}

using DomainLayer.Contracts;
using DomainLayer.Models;
using Mapster;
using Microsoft.AspNetCore.Http;
using ServiceAbstractionLayer;
using ServiceLayer.Specifications.AssignmentSpecifications;
using Shared.Dtos.AssigmentDto.Request;
using Shared.Dtos.AssigmentDto.Response;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
	public class AssignmentService(IUnitOfWork unitOfWork, IFileStorageService fileStorageService) : IAssigmentService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;

		public async Task<AssigmentResponseDto> CreateOrUpdateAssigmentForCourse(int courseId, AssigmentRequestDto assigmentRequest)
		{
			var assignmentRepository = _unitOfWork.GetRepository<Assignment, int>();
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			
			// Verify course exists
			var course = await courseRepository.GetByIdAsync(courseId);
			if (course is null)
			{
				throw new Exception($"Course with id {courseId} not found");
			}

			if (assigmentRequest.Id > 0)
			{
				//Update 
				var foundedAssignmentEntity = await assignmentRepository.GetByIdAsync(assigmentRequest.Id);
				if (foundedAssignmentEntity is null)
				{
					throw new Exception($"Assignment with id {assigmentRequest.Id} not found");
				}

				// Update properties on the tracked entity instead of creating a new one
				foundedAssignmentEntity.Title = assigmentRequest.Title;
				foundedAssignmentEntity.Description = assigmentRequest.Description;
				foundedAssignmentEntity.DueDate = assigmentRequest.DueDate;
				foundedAssignmentEntity.TotalMarks = assigmentRequest.TotalMarks;
				foundedAssignmentEntity.CourseId = courseId;

				assignmentRepository.Update(foundedAssignmentEntity);
			}
			else
			{
				//Create
				var assignmentEntity = assigmentRequest.Adapt<Assignment>();
				assignmentEntity.CourseId = courseId;
				await assignmentRepository.AddAsync(assignmentEntity);
			}

			await _unitOfWork.SaveChangesAsync();

			// Reload with attachments for the response
			var assignmentSpec = new AssignmentSpecification(assigmentRequest.Id > 0 ? assigmentRequest.Id : course.Assignments?.LastOrDefault()?.Id ?? 0);
			var updatedAssignment = await assignmentRepository.GetByIdAsync(assignmentSpec);
			return updatedAssignment!.Adapt<AssigmentResponseDto>();
		}

		public async Task<AssigmentResponseDto> AddAttachmentToAssigment(int AssigmentId, List<IFormFile?> Files)
		{
			var assignmentRepository = _unitOfWork.GetRepository<Assignment, int>();
			var attachmentRepository = _unitOfWork.GetRepository<AssignmentAttachment, Guid>();
			
			var assignmentEntity = await assignmentRepository.GetByIdAsync(AssigmentId);
			if (assignmentEntity is null)
			{
				throw new Exception($"Assignment with id {AssigmentId} not found");
			}

			foreach (var file in Files)
			{
				if (file is not null && file.Length > 0)
				{
					using var stream = file.OpenReadStream();
					var fileUrl = await _fileStorageService.UploadFileAsync(
						stream,
						file.FileName,
						$"assignments/{AssigmentId}/attachments",
						file.ContentType
					);

					var attachment = new AssignmentAttachment
					{
						FileName = file.FileName,
						FileUrl = fileUrl,
						Type = file.ContentType,
						AssignmentId = AssigmentId
					};

					await attachmentRepository.AddAsync(attachment);
				}
			}

			await _unitOfWork.SaveChangesAsync();
			
			// Reload with attachments
			var assignmentSpec = new AssignmentSpecification(AssigmentId);
			var updatedAssignment = await assignmentRepository.GetByIdAsync(assignmentSpec);
			return updatedAssignment!.Adapt<AssigmentResponseDto>();
		}

		public async Task DeleteAssigmentAsync(int AssigmentId)
		{
			var assignmentRepository = _unitOfWork.GetRepository<Assignment, int>();
			var assignmentEntity = await assignmentRepository.GetByIdAsync(AssigmentId);
			if (assignmentEntity is null)
			{
				throw new Exception($"Assignment with id {AssigmentId} not found");
			}
			assignmentRepository.Delete(assignmentEntity!);
			await _unitOfWork.SaveChangesAsync();
		}

		public async Task<IEnumerable<AssigmentResponseDto>> GetAllAssigmentsByCourseIdAsync(int courseId)
		{
			var assignmentRepository = _unitOfWork.GetRepository<Assignment, int>();
			var assignmentSpecification = new AssignmentByCourseIdSpecification(courseId);
			var assignmentEntities = await assignmentRepository.GetAllAsync(assignmentSpecification);
			return assignmentEntities.Adapt<IEnumerable<AssigmentResponseDto>>();
		}

		public async Task<AssigmentResponseDto> GetAssigmentByIdAsync(int AssigmentId)
		{
			var assignmentRepository = _unitOfWork.GetRepository<Assignment, int>();
			var assignmentSpec = new AssignmentSpecification(AssigmentId);
			var assignmentEntity = await assignmentRepository.GetByIdAsync(assignmentSpec);
			if (assignmentEntity is null)
			{
				throw new Exception($"Assignment with id {AssigmentId} not found");
			}
			return assignmentEntity.Adapt<AssigmentResponseDto>();
		}

		public async Task RemoveAttachment(Guid AttachmentId)
		{
			var attachmentRepository = _unitOfWork.GetRepository<AssignmentAttachment, Guid>();
			var attachmentEntity = await attachmentRepository.GetByIdAsync(AttachmentId);
			if (attachmentEntity is null)
			{
				throw new Exception($"Attachment with id {AttachmentId} not found");
			}
			attachmentRepository.Delete(attachmentEntity!);
			await _unitOfWork.SaveChangesAsync();
		}
	}
}
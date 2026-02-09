using DomainLayer.Contracts;
using DomainLayer.Exceptions;
using DomainLayer.Models;
using Mapster;
using Microsoft.AspNetCore.Http;
using ServiceAbstractionLayer;
using ServiceLayer.Specifications.ContentSpecifications;
using Shared.Dtos.ContentDto.ContentRequest;
using Shared.Dtos.ContentDto.ContentResponse;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Services
{
	public class ContentService(IUnitOfWork unitOfWork , IFileStorageService fileStorageService) : IContentService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly IFileStorageService _fileStorageService = fileStorageService;
		public async Task<ContentResponseDto> AddAttachmentToContent(int ContentId, List<IFormFile?> Files)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var attachmentRepository = _unitOfWork.GetRepository<ContentAttachment, Guid>();
			var contentEntity = await contentRepository.GetByIdAsync(ContentId);
			if (contentEntity is null)
			{
				throw new ContentNotFoundException(ContentId);
			}
			foreach (var file in Files)
			{
				if (file is not null && file.Length > 0)
				{
					using var stream = file.OpenReadStream();
					var fileUrl = await _fileStorageService.UploadFileAsync(
						stream,
						file.FileName,
						$"contents/{ContentId}/attachments",
						file.ContentType
					);
					var attachment = new ContentAttachment
					{
						FileName = file.FileName,
						FileUrl = fileUrl,
						ContentType = file.ContentType,
						ContentId = ContentId
					};
					await attachmentRepository.AddAsync(attachment);
				}
			}
			await _unitOfWork.SaveChangesAsync();
			// Reload with attachments
			var contentSpec = new ContentSpecification(ContentId);
			var updatedContent = await contentRepository.GetByIdAsync(contentSpec);
			return updatedContent!.Adapt<ContentResponseDto>();
		}

		public async Task<ContentResponseDto> CreateOrUpdateContentForCourse(int courseId, ContentRequestDto contentRequest)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var courseRepository = _unitOfWork.GetRepository<Course, int>();
			var course = await courseRepository.GetByIdAsync(courseId);
			if (course is null)
			{
				throw new CourseNotFoundException(courseId);
			}
			
			if (contentRequest.Id > 0) 
			{
				//Update 
				var foundedContentEntity = await contentRepository.GetByIdAsync(contentRequest.Id);
				if (foundedContentEntity is null)
				{
					throw new ContentNotFoundException(contentRequest.Id);
				}
				
				// Update properties on the tracked entity instead of creating a new one
				foundedContentEntity.Title = contentRequest.Title;
				foundedContentEntity.Body = contentRequest.Body;
				foundedContentEntity.CourseId = courseId;
				
				contentRepository.Update(foundedContentEntity);
			}
			else 
			{
				//create
				var contentEntity = contentRequest.Adapt<Content>();
				contentEntity.CourseId = courseId;
				await contentRepository.AddAsync(contentEntity);
			}
			
			await _unitOfWork.SaveChangesAsync();
			
			// Reload with attachments for the response
			var contentSpec = new ContentSpecification(contentRequest.Id > 0 ? contentRequest.Id : course.Contents?.LastOrDefault()?.Id ?? 0);
			var updatedContent = await contentRepository.GetByIdAsync(contentSpec);
			return updatedContent!.Adapt<ContentResponseDto>();
		}

		public async Task DeleteContentAsync(int contentId)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var contentEntity = await contentRepository.GetByIdAsync(contentId);
			if (contentEntity is null)
			{
				throw new ContentNotFoundException(contentId);
			}
			contentRepository.Delete(contentEntity!);
			await _unitOfWork.SaveChangesAsync();	
		}

		public async Task<IEnumerable<ContentResponseDto>> GetAllContentsByCourseIdAsync(int courseId)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var contentSpecification = new ContentByCourseIdSpecification(courseId);
			var ContentEntities = await contentRepository.GetAllAsync(contentSpecification);
			if (ContentEntities is null || !ContentEntities.Any())
			{
				throw new ContentsInCourseNotFoundException(courseId);
            }
            return ContentEntities.Adapt<IEnumerable<ContentResponseDto>>();
		}

		public async Task<ContentResponseDto> GetContentByIdAsync(int contentId)
		{
			var contentRepository = _unitOfWork.GetRepository<Content, int>();
			var contentSpec = new ContentSpecification(contentId);
			var contentEntity = await contentRepository.GetByIdAsync(contentSpec);
			if (contentEntity is null)
			{
				throw new ContentNotFoundException(contentId);
			}
			return contentEntity.Adapt<ContentResponseDto>();
		}

		public async Task RemoveAttachment(Guid AttachmentId)
		{
			var attachmentRepository = _unitOfWork.GetRepository<ContentAttachment, Guid>();
			 var attachmentEntity = await attachmentRepository.GetByIdAsync(AttachmentId);
			if (attachmentEntity is null)
			{
				throw new ContentAttachmentNotFoundException(AttachmentId);
			}
			attachmentRepository.Delete(attachmentEntity!);
			await _unitOfWork.SaveChangesAsync();
		}
	}
}

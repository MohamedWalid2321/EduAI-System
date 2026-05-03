using DomainLayer.Contracts;
using DomainLayer.Models;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using ServiceAbstractionLayer;
using Shared.Constants;

namespace ServiceLayer.Jobs
{
	public class AssignmentDeadlineReminderJob(
		IUnitOfWork unitOfWork,
		UserManager<ApplicationUser> userManager,
		INotificationService notificationService)
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;
		private readonly UserManager<ApplicationUser> _userManager = userManager;
		private readonly INotificationService _notificationService = notificationService;

		public async Task SendDeadlineRemindersAsync()
		{
			var now = DateTime.Now;
			var windowEnd = now.AddHours(24);

			// Get all assignments due within the next 24 hours
			var assignmentRepo = _unitOfWork.GetRepository<Assignment, int>();
			var upcomingAssignments = (await assignmentRepo.GetAllAsync())
				.Where(a => a.DueDate > now && a.DueDate <= windowEnd)
				.ToList();

			if (upcomingAssignments.Count == 0)
				return;

			// Get all users who have the SolveAss permission (students)
			var allUsers = _userManager.Users.ToList();
			var students = new List<ApplicationUser>();

			foreach (var user in allUsers)
			{
				var claims = await _userManager.GetClaimsAsync(user);
				if (claims.Any(c => c.Type == Permissions.Type && c.Value == Permissions.SolveAss))
					students.Add(user);
			}

			if (students.Count == 0)
				return;

			// Send a notification to each student for each upcoming assignment
			foreach (var assignment in upcomingAssignments)
			{
				var hoursLeft = (int)(assignment.DueDate - now).TotalHours;
				var title = "Assignment Deadline Reminder";
				var message = $"The assignment \"{assignment.Title}\" is due in approximately {hoursLeft} hour(s). Make sure to submit before the deadline.";

				foreach (var student in students)
				{
					await _notificationService.CreateNotificationAsync(student.Id, title, message);
				}
			}
		}
	}
}
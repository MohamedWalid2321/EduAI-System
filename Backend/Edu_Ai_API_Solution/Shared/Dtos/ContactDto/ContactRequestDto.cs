using System.ComponentModel.DataAnnotations;

namespace Shared.Dtos.ContactDto
{
    public record ContactRequestDto
    {
        [Required(ErrorMessage = "Full name is required.")]
        [MaxLength(150, ErrorMessage = "Full name cannot exceed 150 characters.")]
        public string FullName { get; init; } = string.Empty;

        [Required(ErrorMessage = "Email address is required.")]
        [EmailAddress(ErrorMessage = "Invalid email address format.")]
        public string EmailAddress { get; init; } = string.Empty;

        [Required(ErrorMessage = "Subject is required.")]
        [MaxLength(250, ErrorMessage = "Subject cannot exceed 250 characters.")]
        public string Subject { get; init; } = string.Empty;

        [Required(ErrorMessage = "Message content is required.")]
        [MaxLength(5000, ErrorMessage = "Message content cannot exceed 5000 characters.")]
        public string MessageContent { get; init; } = string.Empty;
    }
}

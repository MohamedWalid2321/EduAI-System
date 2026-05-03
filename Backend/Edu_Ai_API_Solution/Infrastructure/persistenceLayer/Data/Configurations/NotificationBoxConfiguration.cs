using DomainLayer.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace persistenceLayer.Data.Configurations
{
	public class NotificationBoxConfiguration : IEntityTypeConfiguration<NotificationBox>
	{
		public void Configure(EntityTypeBuilder<NotificationBox> builder)
		{
			builder.HasKey(nb => nb.Id);

			builder.HasOne(nb => nb.User)
				.WithOne(u => u.NotificationBox)
				.HasForeignKey<NotificationBox>(nb => nb.UserId)
				.OnDelete(DeleteBehavior.Cascade);

			builder.HasMany(nb => nb.Notifications)
				.WithOne(n => n.NotificationBox)
				.HasForeignKey(n => n.NotificationBoxId)
				.OnDelete(DeleteBehavior.Cascade);
		}
	}
}
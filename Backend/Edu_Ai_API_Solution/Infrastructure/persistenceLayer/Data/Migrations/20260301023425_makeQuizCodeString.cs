using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class makeQuizCodeString : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AlterColumn<string>(
                name: "QuizCode",
                table: "Quizzes",
                type: "nvarchar(450)",
                nullable: false,
                oldClrType: typeof(int),
                oldType: "int");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEJ1Qwo4srpKLyZSxh+GAecQzdxOruj2BS7JlB/usLTuN86+ndanaJljejVg5WXZCrw==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEBXPmv6rfHJDgDTYmaTbTRTsIS+Adcz5WZmQb+SO4XPjkuKgesjuq/geoUavEkKSqQ==");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AlterColumn<int>(
                name: "QuizCode",
                table: "Quizzes",
                type: "int",
                nullable: false,
                oldClrType: typeof(string),
                oldType: "nvarchar(450)");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEI8lL+1b8QQssH/g8bPb1WzbXknIp14CxyCoL7OQK6946VU9c8uAxUgLsQ7LQ9sTYg==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEC2KYt4ecW0x5EMAgZAbJtRR4COk4OxUVwyfl/eiQya0x/Ajx41WYMKcamAyV/P27w==");
        }
    }
}

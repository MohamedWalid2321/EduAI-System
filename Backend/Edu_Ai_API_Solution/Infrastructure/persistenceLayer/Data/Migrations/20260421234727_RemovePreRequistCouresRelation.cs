using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class RemovePreRequistCouresRelation : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Courses_Courses_PrerequisiteCourseId",
                table: "Courses");

            migrationBuilder.DropIndex(
                name: "IX_Courses_PrerequisiteCourseId",
                table: "Courses");

            migrationBuilder.DropColumn(
                name: "PrerequisiteCourseId",
                table: "Courses");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEJ1lIaghDNNualwBRUIDehdAwM63E6kkMaV/MNCY57avT5iR2DkzjmqZCWrm22n1LQ==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEHhi/XxOdbVlzNMz77Rl3mBn133oJ7VNqit8hI9sZehJ0MYrbNa+vMHacNOAgwzTfw==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5722));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5767));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5772));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5769));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 22, 1, 47, 26, 587, DateTimeKind.Local).AddTicks(5770));
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "PrerequisiteCourseId",
                table: "Courses",
                type: "int",
                nullable: true);

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEMt9ZJ4g5jJbKroRb86bqTCLOaTEhkS1KfhvqjtSlxf4t1/aAk0EbC/exQSh7sovaA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEEfbGPcbdhufhwNBfhRLqbITBk24xYBI2z97IxGoaBAeskSwHOb0brqKK+FafMq1vw==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8063));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8132));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8141));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8136));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 10, 5, 4, 9, 744, DateTimeKind.Local).AddTicks(8139));

            migrationBuilder.CreateIndex(
                name: "IX_Courses_PrerequisiteCourseId",
                table: "Courses",
                column: "PrerequisiteCourseId");

            migrationBuilder.AddForeignKey(
                name: "FK_Courses_Courses_PrerequisiteCourseId",
                table: "Courses",
                column: "PrerequisiteCourseId",
                principalTable: "Courses",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);
        }
    }
}
